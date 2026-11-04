"""Checkpoint storage beyond this machine, so a run survives the end of a session and can
continue on another platform.

    Run(storage="hf://user/repo")                       # private Hugging Face repo
    Run(storage="/content/drive/MyDrive/my_run")        # a mounted Google Drive folder
    Run(storage=HuggingFaceHub("user/repo", path="exp1"))

No torch imports here.
"""

import os

from .base import MANIFEST, Storage, StorageError, sha256_file
from .hub import HuggingFaceHub
from .local import LocalFolder
from .sync import RemoteCheckpoints


def open_storage(spec):
    """A Storage from "hf://user/repo[/folder]", a folder path or a Storage object."""
    if isinstance(spec, Storage):
        return spec
    if isinstance(spec, str) and spec.startswith("hf://"):
        parts = spec[len("hf://"):].strip("/").split("/")
        if len(parts) < 2 or not all(parts[:2]):
            raise ValueError('Hugging Face storage looks like "hf://<user>/<repo-name>"')
        return HuggingFaceHub("/".join(parts[:2]), path="/".join(parts[2:]))
    if isinstance(spec, (str, os.PathLike)):
        if "://" in os.fspath(spec):
            raise ValueError("storage must be hf://<user>/<repo>, a folder path (such as a "
                             "mounted Google Drive folder) or a Storage object")
        return LocalFolder(spec)
    raise TypeError(f"storage must be a string, a path or a Storage object, not "
                    f"{type(spec).__name__}")


__all__ = ["Storage", "StorageError", "LocalFolder", "HuggingFaceHub", "RemoteCheckpoints",
           "open_storage", "sha256_file", "MANIFEST"]
