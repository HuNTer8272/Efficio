"""Storage in any folder, e.g. a mounted Google Drive folder on Colab."""

import json
import os
import shutil

from .base import MANIFEST, Storage, StorageError

DRIVE_ROOT = "/content/drive"
DRIVE_NOT_MOUNTED = (
    "Google Drive is not mounted, so {path} would be on the session's own disk and lost "
    "when the session ends. Mount Drive first (Efficio doesn't do it for you):\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')")


def _replace(src, dst):
    try:
        os.replace(src, dst)
    except OSError:              # some network/FUSE folders can't rename over a file
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)


class LocalFolder(Storage):
    """Checkpoints copied to a folder: a Google Drive folder (/content/drive/MyDrive/...)
    after you mount Drive, a network share or a second disk. Every copy goes to a temporary
    name first and is then renamed, so a half-copied file is never seen under its name."""

    def __init__(self, path):
        raw = os.fspath(path)
        self._on_drive = raw.replace("\\", "/").startswith(DRIVE_ROOT + "/")
        self._raw = raw
        self.path = os.path.abspath(os.path.expanduser(raw))

    def describe(self):
        return self._raw if self._on_drive else self.path

    def connect(self, log=print):
        if self._on_drive and not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
            raise StorageError(DRIVE_NOT_MOUNTED.format(path=self._raw))
        os.makedirs(self.path, exist_ok=True)

    def _file(self, name):
        if not name or os.path.basename(name) != name or name in (".", ".."):
            raise ValueError(f"not a plain file name: {name!r}")
        return os.path.join(self.path, name)

    def upload(self, local_path, name):
        final = self._file(name)
        partial = final + ".partial"
        shutil.copyfile(local_path, partial)
        _replace(partial, final)

    def download(self, name, local_path):
        shutil.copyfile(self._file(name), local_path)

    def list(self):
        if not os.path.isdir(self.path):
            return []
        return sorted(n for n in os.listdir(self.path)
                      if not n.endswith(".partial") and os.path.isfile(os.path.join(self.path, n)))

    def delete(self, name):
        try:
            os.remove(self._file(name))
        except FileNotFoundError:
            pass

    def read_manifest(self):
        try:
            with open(self._file(MANIFEST), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def write_manifest(self, manifest):
        final = self._file(MANIFEST)
        partial = final + ".partial"
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        _replace(partial, final)
