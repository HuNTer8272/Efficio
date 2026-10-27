"""Keep copies of the local checkpoints in a storage backend, and get them back.

After each local save the file is uploaded in a background thread (one upload at a time),
with retries. Only after the upload has finished is it added to the manifest, so a
half-uploaded file is never the latest. Then checkpoints beyond the newest `keep` are
deleted. An upload error is kept and raised in the main thread by the next check().

Downloading picks the newest manifest entry and verifies its sha256 before it is used.
"""

import datetime
import os
import re
import threading
import time

from .base import MANIFEST_FORMAT, StorageError, sha256_file

CHECKPOINT_NAME = re.compile(r"^ckpt_step\d+\.pt$")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RemoteCheckpoints:
    """Uploads, retention and verified downloads of checkpoints in one storage backend."""

    def __init__(self, storage, keep=3, platform="local", retries=3, backoff=2.0):
        if keep < 1:
            raise ValueError(f"keep_remote must be at least 1, not {keep!r}")
        self.storage = storage
        self.keep = keep
        self.platform = platform
        self.retries = retries
        self.backoff = backoff
        self.stats = {"uploads": 0, "upload_s": 0.0, "waited_s": 0.0, "last_file": None,
                      "last_bytes": None, "last_time": None, "last_ok": None,
                      "last_error": None, "cleanup_errors": 0, "downloaded": None}
        self._thread = None
        self._current = None
        self._error = None

    # ---------- uploading ----------
    def _retry(self, what, fn, *args):
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args)
            except Exception as e:
                if attempt == self.retries:
                    raise StorageError(f"{what} failed after {attempt} attempts "
                                       f"({type(e).__name__}: {self.storage.redact(e)})") from None
                time.sleep(self.backoff * 2 ** (attempt - 1))

    def upload(self, local_path, info, after=None):
        """Upload a local checkpoint in the background. `info`: step, epoch and the like
        for the manifest. `after`: a thread to wait for first (a background local save).
        Waits for the previous upload and raises its error, if it failed."""
        self.wait()
        self.check()
        self._current = os.path.basename(local_path)
        self._thread = threading.Thread(target=self._upload, args=(local_path, dict(info), after),
                                        name="efficio-upload")
        self._thread.start()

    def _upload(self, local_path, info, after):
        start = time.perf_counter()
        name = os.path.basename(local_path)
        try:
            if after is not None:
                after.join()
            if not os.path.exists(local_path):
                raise StorageError(f"{local_path} was not written, so it was not uploaded")
            size, digest = os.path.getsize(local_path), sha256_file(local_path)
            self._retry(f"uploading {name}", self.storage.upload, local_path, name)
            entry = {"file": name, **info, "size": size, "sha256": digest,
                     "platform": self.platform, "time": _now()}
            manifest = self._retry("reading the manifest", self.storage.read_manifest) or {}
            entries = [e for e in manifest.get("checkpoints", []) if e.get("file") != name]
            entries.append(entry)
            kept = entries[-self.keep:]
            # The new file counts only once the manifest names it; older files go after that.
            self._retry("writing the manifest", self.storage.write_manifest,
                        {"format": MANIFEST_FORMAT, "checkpoints": kept})
            self.stats.update(uploads=self.stats["uploads"] + 1, last_file=name,
                              last_bytes=size, last_time=entry["time"], last_ok=True,
                              last_error=None)
            self._remove_old({e["file"] for e in kept})
        except BaseException as e:                  # kept and raised in the main thread
            if not isinstance(e, StorageError):
                e = StorageError(f"uploading {name} failed "
                                 f"({type(e).__name__}: {self.storage.redact(e)})")
            self._error = e
            self.stats.update(last_ok=False, last_error=str(e))
        finally:
            self.stats["upload_s"] += time.perf_counter() - start

    def _remove_old(self, kept):
        """Delete checkpoints not in the manifest: older ones, and any left by an upload whose
        manifest update failed. A failed delete only leaves an extra file behind."""
        try:
            names = self.storage.list()
        except Exception:
            self.stats["cleanup_errors"] += 1
            return
        for name in names:
            if CHECKPOINT_NAME.match(name) and name not in kept:
                try:
                    self.storage.delete(name)
                except Exception:
                    self.stats["cleanup_errors"] += 1

    @property
    def pending(self):
        """True while an upload is running."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout=None):
        """Wait for the running upload. Returns False if it is still running after
        `timeout` seconds. Doesn't raise; see check()."""
        if self._thread is not None:
            if self._thread.is_alive():
                start = time.perf_counter()
                self._thread.join(timeout)
                self.stats["waited_s"] += time.perf_counter() - start
            if self._thread.is_alive():
                return False
            self._thread = None
        return True

    def check(self):
        """Raise the error of a finished upload, if there was one. Doesn't wait."""
        if self._error is not None and not self.pending:
            error, self._error = self._error, None
            raise RuntimeError(f"Uploading a checkpoint to {self.storage.describe()} failed: "
                               f"{error}. The local checkpoint is fine; fix the problem and "
                               "start again to resume from it.") from None

    def finish(self, timeout=None):
        """Wait for the running upload and say how it went, without raising:
        ("ok" | "failed" | "running" | "none", message)."""
        name = self._current
        if not self.wait(timeout):
            return "running", (f"{name} is still uploading after {timeout:.0f} s; it goes on "
                               "in the background, keep this session open until it finishes")
        if self._error is not None:
            error, self._error = self._error, None
            return "failed", str(error)
        if name is None:
            return "none", "nothing was uploaded"
        size = self.stats["last_bytes"] or 0
        return "ok", f"{name} ({size / 1024**2:.1f} MB) is in {self.storage.describe()}"

    # ---------- resuming ----------
    def latest(self):
        """The newest checkpoint entry in the manifest, or None."""
        manifest = self._retry("reading the manifest", self.storage.read_manifest) or {}
        entries = [e for e in manifest.get("checkpoints", [])
                   if CHECKPOINT_NAME.match(str(e.get("file", ""))) and e.get("sha256")]
        return entries[-1] if entries else None

    def download(self, entry, directory):
        """Download a manifest entry into `directory`, verify its sha256 and return the path.
        Raises StorageError (and keeps nothing) if the file doesn't match."""
        name = entry["file"]
        if not CHECKPOINT_NAME.match(name):
            raise StorageError(f"unexpected checkpoint name in the manifest: {name!r}")
        os.makedirs(directory, exist_ok=True)
        final = os.path.join(directory, name)
        tmp = final + ".download"
        try:
            self._retry(f"downloading {name}", self.storage.download, name, tmp)
            digest = sha256_file(tmp)
            if digest != entry["sha256"]:
                raise StorageError(
                    f"{name} from {self.storage.describe()} doesn't match its sha256 in the "
                    f"manifest (expected {entry['sha256'][:16]}..., got {digest[:16]}...), so "
                    "it was not used. The file is damaged or was changed; delete it and its "
                    "manifest entry, or start without resume.")
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.stats["downloaded"] = dict(entry)
        return final
