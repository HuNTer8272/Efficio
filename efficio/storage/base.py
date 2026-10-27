"""The storage interface: a place outside this machine where checkpoints are kept.

A backend implements five file operations (upload, download, list, delete, and connect to
check access). The manifest helpers are built on them, and backends may override them.
Nothing here imports torch, so other frameworks can reuse it.
"""

import hashlib
import json
import os
import re
import tempfile

MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{10,}")


class StorageError(RuntimeError):
    """A storage operation failed. The message never contains a token."""


def redact(text, secrets=()):
    """Remove secrets (and anything shaped like a Hugging Face token) from a message."""
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<hidden>")
    return _HF_TOKEN_PATTERN.sub("hf_<hidden>", text)


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Storage:
    """Base class for checkpoint storage. Names are plain file names (no folders)."""

    def describe(self):
        """Where the files go, for logs and the report. Never includes a token."""
        return type(self).__name__

    def connect(self, log=print):
        """Check access before training starts (and create the target if needed)."""

    def upload(self, local_path, name):
        raise NotImplementedError

    def download(self, name, local_path):
        raise NotImplementedError

    def list(self):
        """Names of the files in the storage."""
        raise NotImplementedError

    def delete(self, name):
        raise NotImplementedError

    def redact(self, text):
        """A message with this backend's secrets removed."""
        return redact(text)

    def read_manifest(self):
        """The manifest dict, or None if there is none yet."""
        if MANIFEST not in self.list():
            return None
        with tempfile.TemporaryDirectory(prefix="efficio-manifest-") as tmp:
            path = os.path.join(tmp, MANIFEST)
            self.download(MANIFEST, path)
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    def write_manifest(self, manifest):
        with tempfile.TemporaryDirectory(prefix="efficio-manifest-") as tmp:
            path = os.path.join(tmp, MANIFEST)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            self.upload(path, MANIFEST)

    def __repr__(self):
        return f"{type(self).__name__}({self.describe()!r})"
