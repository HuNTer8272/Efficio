"""The checkpoint side of a training run, shared by efficio.Run and the planned framework
adapters: which checkpoint to resume from (on this machine or in storage), saving and
uploading, the time-budget and shutdown-signal stop decision, auto-shutdown and the
checkpoint lines of the report.

The session never runs a training step itself. Whoever owns the loop (Run.loop(), or a
framework callback) calls poll() after each optimizer step and saves or stops as told.
"""

import os
import signal
import time

from ..budget import shutdown
from .manager import (
    DifferentRunError,
    StorageInUseError,
    _efficio_version,
    checkpoint_step,
)


def size_text(nbytes):
    """A byte count as text, e.g. "1.5 MB"."""
    for unit, factor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if nbytes >= factor:
            return f"{nbytes / factor:.1f} {unit}"
    return f"{nbytes} B"


def open_remote(spec, keep, checkpoint_dir, platform, log):
    """Open and connect the storage given as Run(storage=...): "hf://user/repo", a folder
    path or a Storage object. Returns a RemoteCheckpoints that uploads in the background."""
    from ..storage import LocalFolder, RemoteCheckpoints, open_storage

    backend = open_storage(spec)
    if isinstance(backend, LocalFolder) and os.path.abspath(backend.path) == os.path.abspath(
        checkpoint_dir
    ):
        raise ValueError(
            f"storage ({backend.path}) must be a different folder from "
            "checkpoint_dir; it holds copies of the checkpoints there"
        )
    backend.connect(log)
    return RemoteCheckpoints(backend, keep=keep, platform=platform)


class CheckpointSession:
    """Resume, save, upload and stop decisions for one run.

    ckpt: the CheckpointManager; its fingerprint must be set before check_storage_owner()
        and load_latest().
    remote: a RemoteCheckpoints (see open_remote()) or None.
    budget: the interfaces.Budget; poll() stops once it is nearly used.
    resume: "auto", True or False, as in Run: a checkpoint of a different run is skipped
        with a warning ("auto") or raises (True).
    overwrite_storage, upload_timeout: as in Run.
    save_every: seconds between periodic saves.
    log: prints progress messages (warnings are always printed).
    """

    def __init__(
        self,
        ckpt,
        remote=None,
        budget=None,
        resume="auto",
        overwrite_storage=False,
        upload_timeout=120,
        save_every=15 * 60,
        log=print,
    ):
        self.ckpt = ckpt
        self.remote = remote
        self.budget = budget
        self.resume = resume
        self.overwrite_storage = overwrite_storage
        self.upload_timeout = upload_timeout
        self.save_every = save_every
        self.log = log
        self.frozen_tensors = 0  # frozen parameter tensors of the model (upload info)
        self.final_upload = None  # (path, status) of the last waited-for upload
        self.saves = 0
        self.last_save = time.time()
        self.stop_reason = None  # set by poll(), or by the loop owner ("finished")
        self.signal_received = False

    # ---------- resume ----------
    def resume_file(self):
        """The local checkpoint, or the latest one in storage (downloaded) if that is newer,
        e.g. in a new session or on another platform."""
        local = self.ckpt.latest()
        if self.remote is None:
            return local
        from ..storage import StorageError

        where = self.remote.storage.describe()
        try:
            entry = self.remote.latest()
        except StorageError as e:
            if local is None:
                raise StorageError(
                    f"Could not read the checkpoint list in {where}: {e}. "
                    "Nothing was changed there. Check the connection and "
                    "token, or pass resume=False to start fresh."
                ) from None
            self.log(f"[efficio] could not read {where} ({e}); resuming from this machine")
            return local
        if entry is None or (local is not None and checkpoint_step(local) >= entry["step"]):
            return local
        if self.other_run(entry):  # only with overwrite_storage=True; never resumed
            return local
        if local is not None:
            self.log(
                f"[efficio] {where} has a newer checkpoint (step {entry['step']}) than "
                f"this machine (step {checkpoint_step(local)})"
            )
        path = self.remote.download(entry, self.ckpt.directory)
        self.log(
            f"[efficio] downloaded {entry['file']} from {where}: step {entry['step']}, "
            f"{entry.get('mode', 'exact')} mode, saved on {entry.get('platform', '?')} "
            f"at {entry.get('time', '?')}; sha256 verified"
        )
        return path

    def load_latest(self, model, optimizer, scheduler=None, scaler=None):
        """Resume from resume_file(): returns the saved train_state dict, or None when
        there is nothing to resume or the checkpoint belongs to a different run (see
        handle_different_run())."""
        path = self.resume_file()
        try:
            return self.ckpt.load(
                model,
                optimizer,
                scheduler,
                scaler,
                path=path,
                map_location="cpu",
                fingerprint=self.ckpt.fingerprint,
            )
        except DifferentRunError as e:
            return self.handle_different_run(e)

    def other_run(self, entry):
        """True for a manifest entry saved by a different run (entries from before the
        fingerprint have none and count as this run's)."""
        saved = entry.get("fingerprint")
        return bool(saved) and saved.get("hash") != self.ckpt.fingerprint["hash"]

    def check_storage_owner(self):
        """Refuse to write into storage that holds another run's checkpoints, unless
        overwrite_storage=True. Runs before anything is written there."""
        from ..storage import StorageError

        where = self.remote.storage.describe()
        try:
            others = [e for e in self.remote.entries() if self.other_run(e)]
        except StorageError as e:
            raise StorageError(
                f"Could not read the checkpoint list in {where}: {e}. Nothing "
                "was changed there; check the connection and token."
            ) from None
        if not others:
            return
        newest = others[-1]
        if not self.overwrite_storage:
            raise StorageInUseError(
                f"{where} (step {newest['step']})", newest["fingerprint"], self.ckpt.fingerprint
            )
        self.remote.replace_fingerprints = {e["fingerprint"]["hash"] for e in others}
        print(
            f"[efficio] WARNING: overwrite_storage=True: {where} holds {len(others)} "
            f"checkpoint(s) of a different run (newest at step {newest['step']}). They are "
            "not resumed, and are deleted once this run's first checkpoint is uploaded."
        )

    def handle_different_run(self, error):
        """A checkpoint of another run: raise with resume=True; otherwise warn, move the
        local checkpoints aside and return None (start fresh)."""
        self.warn_different_run(error)
        return self.move_aside_local_checkpoints()

    def warn_different_run(self, error):
        """A checkpoint from another run: raise with resume=True, else warn and go on."""
        if self.resume is True:
            raise error
        print(
            f"[efficio] WARNING: not resuming from {error.where}: it was saved by a different "
            f"run ({error.saved.get('optimizer')}, {error.saved.get('parameters')} parameter "
            f"tensors; this run: {error.current['optimizer']}, "
            f"{error.current['parameters']}). Starting fresh. New checkpoints go to the same "
            "places and replace the old ones there; use another checkpoint_dir or storage "
            "to keep them, or pass resume=True to make this an error."
        )

    def move_aside_local_checkpoints(self):
        """Starting fresh next to another run's checkpoints: move them into a subfolder, so
        they are neither resumed nor mixed up with this run's (their steps may be higher).
        Returns None (nothing resumed)."""
        files = [
            f
            for f in os.listdir(self.ckpt.directory)
            if f.startswith("ckpt_step") and f.endswith(".pt")
        ]
        aside = os.path.join(self.ckpt.directory, time.strftime("other_run_%Y%m%d_%H%M%S"))
        os.makedirs(aside, exist_ok=True)
        for f in files:
            os.replace(os.path.join(self.ckpt.directory, f), os.path.join(aside, f))
        print(f"[efficio] moved {len(files)} checkpoint file(s) of the other run to {aside}")
        return None

    # ---------- saving and uploading ----------
    def check(self):
        """Raise a failed background save or upload now, not later."""
        self.ckpt.check()
        if self.remote is not None:
            self.remote.check()

    def save(
        self,
        train_state,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        extra=None,
        blocking=True,
    ):
        """Save a checkpoint and, with storage, upload it. Returns the path. A blocking save
        also waits for the upload (up to upload_timeout seconds) and says whether it
        succeeded; with blocking=False both run in the background."""
        if self.remote is not None:
            self.remote.wait()  # one upload at a time; its file must stay until done
        path = self.ckpt.save(
            train_state,
            model,
            optimizer,
            scheduler,
            scaler,
            extra=extra,
            blocking=blocking,
        )
        self.last_save = time.time()
        self.saves += 1
        if self.ckpt.pending:
            self.log(f"[efficio] checkpoint saving in the background: {path}")
        else:
            self.log(f"[efficio] checkpoint saved: {path}")
        if self.remote is not None:
            self.upload(path, train_state, blocking)
        return path

    def upload(self, path, train_state, blocking):
        """Upload a saved checkpoint; a blocking upload is waited for and its status kept
        in final_upload."""
        info = {
            "step": train_state.step,
            "epoch": train_state.epoch,
            "batch_in_epoch": train_state.batch_in_epoch,
            "mode": self.ckpt.mode,
            "trainable_only": bool(self.ckpt.trainable_only and self.frozen_tensors),
            "fingerprint": self.ckpt.fingerprint,
            "efficio_version": _efficio_version(),
        }
        if not blocking:
            self.remote.upload(path, info, after=self.ckpt.writer)  # raises an earlier failure
            return
        status, message = self.remote.finish()  # an earlier upload: report, don't raise
        if status == "failed":
            print(f"[efficio] an earlier upload failed: {message}")
        self.remote.upload(path, info)
        status, message = self.remote.finish(self.upload_timeout)
        self.final_upload = (path, status)
        if status == "ok":
            self.log(f"[efficio] upload complete: {message}")
        elif status == "failed":
            print(
                f"[efficio] UPLOAD FAILED: {message}\n[efficio] the checkpoint is only on "
                f"this machine: {path}"
            )
        else:
            print(f"[efficio] upload NOT confirmed: {message}")

    # ---------- budget and shutdown signal ----------
    def poll(self):
        """Call after each optimizer step. Returns "stop" (save with blocking=True, then
        stop training), "save" (a periodic save, blocking=False) or None."""
        if self.signal_received:
            self.stop_reason = "shutdown signal"
        elif self.budget.nearly_out():
            self.stop_reason = "time budget nearly used"
        if self.stop_reason:
            self.log(f"[efficio] stopping: {self.stop_reason}")
            return "stop"
        if time.time() - self.last_save >= self.save_every:
            return "save"
        return None

    @property
    def should_stop(self):
        """True when the run stopped early (time budget or shutdown signal)."""
        return self.stop_reason is not None and self.stop_reason != "finished"

    def install_signal_handler(self):
        """Catch SIGTERM while training. Returns the previous handler, or None if ours
        could not be installed."""

        def handler(signum, frame):
            self.signal_received = True  # the save happens at the next poll()

        try:
            return signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):  # not in the main thread, or unsupported
            return None

    @staticmethod
    def restore_signal_handler(old):
        # Put the previous handler back so that, after training, SIGTERM stops the process
        # (or notebook kernel) as usual instead of only setting a flag nobody reads.
        if old is None:
            return
        try:
            signal.signal(signal.SIGTERM, old)
        except (ValueError, OSError, TypeError):
            pass

    # ---------- end of the run ----------
    def maybe_shut_down(self, platform, enabled):
        """shutdown_when_done: release the Colab runtime, only once training ended and the
        final checkpoint is confirmed in storage."""
        if not enabled:
            return
        if platform != "colab":
            self.log(shutdown.startup_notice(platform, self.remote is not None))
            return
        if self.stop_reason is None:
            print("[efficio] not shutting down: run.loop() has not ended")
            return
        path, status = self.final_upload or (None, None)
        if self.remote is None or status != "ok" or path != self.ckpt.stats["last_path"]:
            print(
                "[efficio] NOT shutting down: the final checkpoint is not confirmed in "
                "storage, and it would be lost with the runtime"
            )
            return
        shutdown.release_colab_runtime(self.log)

    def report_lines(self):
        """The checkpoint and storage lines of the end-of-run report. Waits for a running
        background save, and for a running upload up to upload_timeout seconds."""
        return self._checkpoint_lines() + self._storage_lines()

    def _checkpoint_lines(self):
        try:
            self.ckpt.wait()  # finish a background save before reporting
            failed = ""
        except RuntimeError as e:
            failed = f" (last background save FAILED: {e.__cause__ or e})"
        st = self.ckpt.stats
        kind = self.ckpt.mode + (", trainable parameters only" if self.ckpt.trainable_only else "")
        size = f", last file {size_text(st['last_bytes'])}" if st["last_bytes"] is not None else ""
        lines = [f"  checkpoint mode : {kind}{size}{failed}"]
        if st["saves"]:
            hidden = max(st["background_s"] - st["waited_s"], 0.0)
            line = f"  save time       : {st['blocking_s']:.1f} s blocking training"
            if st["background_s"]:
                line += (
                    f", {st['background_s']:.1f} s written in the background "
                    f"({hidden:.1f} s hidden)"
                )
            lines.append(line)
        return lines

    def _storage_lines(self):
        if self.remote is None:
            return []
        status, message = self.remote.finish(self.upload_timeout)  # a periodic upload
        st = self.remote.stats
        lines = [
            f"  storage         : {self.remote.storage.describe()} "
            f"(keeps the last {self.remote.keep})"
        ]
        if st["downloaded"]:
            d = st["downloaded"]
            lines.append(
                f"  downloaded      : {d['file']} (step {d['step']}, saved on "
                f"{d.get('platform', '?')}), sha256 verified"
            )
        if status == "failed" or st["last_ok"] is False:
            why = message if status == "failed" else st["last_error"]
            lines.append(f"  last upload     : FAILED ({why})")
        elif status == "running":
            lines.append(f"  last upload     : still running ({message})")
        if st["last_file"]:
            lines.append(
                f"  last uploaded   : {st['last_file']}, {size_text(st['last_bytes'])} at "
                f"{st['last_time']}"
            )
        if st["uploads"]:
            hidden = max(st["upload_s"] - st["waited_s"], 0.0)
            lines.append(
                f"  upload time     : {st['upload_s']:.1f} s in the background "
                f"({hidden:.1f} s hidden)"
            )
        return lines
