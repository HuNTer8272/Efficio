"""Save and resume training safely.

Phase 1: local disk, "exact" mode. Light mode and Hugging Face Hub / Drive storage come in Phase 2.
"""

import glob
import os
import random

import torch

try:
    import numpy as np
except ImportError:          # numpy is optional
    np = None


def _rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory, keep_last=2, mode="exact"):
        self.directory = directory
        self.keep_last = keep_last
        self.mode = mode
        os.makedirs(directory, exist_ok=True)

    def _path(self, step):
        return os.path.join(self.directory, f"ckpt_step{step:09d}.pt")

    def save(self, train_state, model, optimizer, scheduler=None, scaler=None):
        payload = {
            "efficio_version": 1,
            "mode": self.mode,
            "train_state": train_state.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _rng_state(),
        }
        final = self._path(train_state.step)
        tmp = final + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, final)        # atomic: a crash mid-save never corrupts the last good file
        self._cleanup()
        return final

    def latest(self):
        files = sorted(glob.glob(os.path.join(self.directory, "ckpt_step*.pt")))
        return files[-1] if files else None

    def load(self, model, optimizer, scheduler=None, scaler=None, path=None, map_location="cpu"):
        """Restore everything. Returns the saved train_state dict, or None if nothing to resume."""
        path = path or self.latest()
        if path is None:
            return None
        # Our own file holds RNG states and plain dicts, so the weights_only loader can't read it.
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        _set_rng_state(payload["rng"])
        return payload["train_state"]

    def _cleanup(self):
        files = sorted(glob.glob(os.path.join(self.directory, "ckpt_step*.pt")))
        for old in files[:-self.keep_last]:
            os.remove(old)
