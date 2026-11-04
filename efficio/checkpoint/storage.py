"""Checkpoint storage backends live in efficio.storage; this module re-exports them."""

from ..storage import (  # noqa: F401
    HuggingFaceHub,
    LocalFolder,
    RemoteCheckpoints,
    Storage,
    StorageError,
    open_storage,
)
