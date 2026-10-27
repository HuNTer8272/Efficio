"""Save and resume training safely.

Three modes (see checkpoint/light.py for the encodings):
- "exact" (default): everything at full precision; a bit-identical resume.
- "light": model weights at full precision, optimizer state compressed to 8 bits. About 2x
  smaller for Adam-trained models. The weights resume exactly; only the optimizer's
  momentum and second-moment estimates are approximate, so the effect on training is tiny.
- "smallest": like light, plus floating-point weights stored in fp16. About 3x smaller; the
  weights are rounded to fp16, a slight effect after a resume.
Any mode loads whatever mode the checkpoint was written in.

With trainable_only, frozen parameters (requires_grad=False) are not saved. A signature of
them (names, shapes, dtypes and a checksum) is saved instead, and loading refuses to resume
if the frozen weights in the model differ from the ones the checkpoint was made with.

With async_save, save(blocking=False) copies the state to CPU right away and writes the file
in a background thread. Every write uses a temp file and an atomic rename, so an interrupted
write never corrupts the latest checkpoint. Only one save runs at a time, and an error in the
background thread is raised in the main thread by the next check(), wait() or save().
"""

import copy
import glob
import hashlib
import json
import math
import os
import random
import threading
import time

import torch

from . import light

try:
    import numpy as np
except ImportError:          # numpy is optional
    np = None

# Format 1: Efficio 0.1.x. Format 2: early 0.2.0 builds, where "light" meant fp16 weights
# plus 8-bit optimizer state (now called "smallest"). Format 3: exact / light / smallest.
FORMAT_VERSION = 3
MODES = ("exact", "light", "smallest")
_SIGNATURE_SAMPLE = 4096     # elements hashed per frozen tensor


class CheckpointMismatchError(RuntimeError):
    """The checkpoint doesn't fit the model, e.g. its frozen weights are different."""


class DifferentRunError(CheckpointMismatchError):
    """The checkpoint was saved by a different run: other parameter names or shapes, or
    another optimizer type."""

    def __init__(self, where, saved, current):
        super().__init__(
            f"{where} was saved by a different run ({describe_fingerprint(saved)}) than this "
            f"one ({describe_fingerprint(current)}): the model's parameter names or shapes, or "
            "the optimizer type, differ. Use another checkpoint_dir or storage for this run.")
        self.where, self.saved, self.current = where, saved, current


def run_fingerprint(model, optimizer):
    """What identifies a run's checkpoints: the model's parameter names and shapes and the
    optimizer type. The checkpoint mode is not part of it."""
    params = [[name, list(p.shape)] for name, p in model.named_parameters()]
    kind = type(optimizer).__name__
    text = json.dumps({"parameters": params, "optimizer": kind})
    return {"hash": hashlib.sha256(text.encode()).hexdigest()[:16], "optimizer": kind,
            "parameters": len(params)}


def describe_fingerprint(fp):
    return f"{fp.get('optimizer', '?')}, {fp.get('parameters', '?')} parameter tensors"


def _efficio_version():
    try:
        from importlib.metadata import version
        return version("efficio")
    except Exception:
        return "unknown"


def _rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    # RNG states must be CPU ByteTensors, even when restoring CUDA generators.
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _to_cpu(obj):
    """A CPU copy of a (nested) state: tensors are copied, everything else deep-copied, so
    training can go on while the copy is written."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def tensor_signature(t):
    """Shape, dtype, total and a hash of evenly spaced elements: cheap to compute, and any
    real change of the weights (another pretrained model, a different init) changes it."""
    t = t.detach()
    flat = t.reshape(-1)
    n = flat.numel()
    if n:
        idx = torch.linspace(0, n - 1, steps=min(n, _SIGNATURE_SAMPLE), device=flat.device).long()
        sample = flat[idx].contiguous().cpu()
        digest = hashlib.sha1(sample.view(torch.uint8).numpy().tobytes()).hexdigest()
        total = float(flat.double().sum().item()) if (flat.is_floating_point() or
                                                      flat.is_complex()) else int(flat.sum())
    else:
        digest, total = "", 0.0
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "sample_sha1": digest, "sum": total}


def checkpoint_step(path):
    """The step in a checkpoint file name (ckpt_step000001234.pt), or -1."""
    digits = os.path.basename(path)[len("ckpt_step"):-len(".pt")]
    return int(digits) if digits.isdigit() else -1


def _same_signature(a, b):
    if (a["shape"], a["dtype"], a["sample_sha1"]) != (b["shape"], b["dtype"], b["sample_sha1"]):
        return False
    return math.isclose(a["sum"], b["sum"], rel_tol=1e-6, abs_tol=1e-6)


class CheckpointManager:
    """Saves and loads full training checkpoints in one folder, keeping the newest
    `keep_last` files. Every save is atomic."""

    def __init__(self, directory, keep_last=2, mode="exact", trainable_only=False,
                 async_save=False):
        if mode not in MODES:
            raise ValueError('checkpoint_mode must be "exact" (default, bit-identical), '
                             '"light" (8-bit optimizer state, about 2x smaller) or "smallest" '
                             f"(also fp16 weights, about 3x smaller), not {mode!r}")
        self.directory = directory
        self.keep_last = keep_last
        self.mode = mode
        self.trainable_only = trainable_only
        self.async_save = async_save
        self.fingerprint = None   # run_fingerprint() of this run, saved in the metadata
        self.extra = {}      # the "extra" dict of the last loaded checkpoint
        self.meta = {}       # the metadata of the last loaded checkpoint ({} for 0.1.x files)
        self.stats = {"saves": 0, "blocking_s": 0.0, "background_s": 0.0, "waited_s": 0.0,
                      "last_path": None, "last_bytes": None}
        self._thread = None
        self._error = None
        os.makedirs(directory, exist_ok=True)

    def _path(self, step):
        return os.path.join(self.directory, f"ckpt_step{step:09d}.pt")

    # ---------- saving ----------
    def save(self, train_state, model, optimizer, scheduler=None, scaler=None, extra=None,
             blocking=True):
        """Save a checkpoint and return its path. `extra`: a small dict of other resume
        state (e.g. the data shuffle seed). With blocking=False and async_save on, the state
        is copied to CPU now and the file is written in the background."""
        start = time.perf_counter()
        self.wait()                                 # one save at a time; raises old errors
        background = self.async_save and not blocking
        payload = self._payload(train_state, model, optimizer, scheduler, scaler, extra,
                                copy_state=background or self.mode != "exact")
        final = self._path(train_state.step)
        if background:
            self._thread = threading.Thread(target=self._write_in_background,
                                            args=(payload, final), name="efficio-checkpoint")
            self._thread.start()
        else:
            self._write(payload, final)
        self.stats["saves"] += 1
        self.stats["blocking_s"] += time.perf_counter() - start
        return final

    def _payload(self, train_state, model, optimizer, scheduler, scaler, extra, copy_state):
        frozen = ({n for n, p in model.named_parameters() if not p.requires_grad}
                  if self.trainable_only else set())
        model_sd = {k: v for k, v in model.state_dict().items() if k not in frozen}
        full_sd = model.state_dict() if frozen else None
        frozen_sig = {k: tensor_signature(full_sd[k]) for k in sorted(frozen) if k in full_sd}
        opt_sd = optimizer.state_dict()
        sched_sd = scheduler.state_dict() if scheduler is not None else None
        scaler_sd = scaler.state_dict() if scaler is not None else None
        if copy_state:                              # a snapshot that training can't change
            model_sd, opt_sd, sched_sd, scaler_sd = (_to_cpu(model_sd), _to_cpu(opt_sd),
                                                     _to_cpu(sched_sd), _to_cpu(scaler_sd))
        dtypes = {}
        if self.mode == "smallest":                 # fp16 weights
            model_sd, dtypes = light.encode_model(model_sd)
        if self.mode in ("light", "smallest"):      # 8-bit optimizer state
            opt_sd = light.encode_optimizer(opt_sd)
        meta = {
            "format_version": FORMAT_VERSION,
            "efficio_version": _efficio_version(),
            "mode": self.mode,
            "trainable_only": bool(frozen),
            "frozen": frozen_sig,
            "model_dtypes": dtypes,
            "fingerprint": self.fingerprint,
            "sizes": {"model_bytes": _tensor_bytes(model_sd),
                      "optimizer_bytes": _tensor_bytes(opt_sd),
                      "frozen_tensors": len(frozen_sig)},
        }
        return {
            "efficio_version": meta["efficio_version"],
            "format_version": FORMAT_VERSION,
            "mode": self.mode,
            "train_state": train_state.to_dict(),
            "model": model_sd,
            "optimizer": opt_sd,
            "scheduler": sched_sd,
            "scaler": scaler_sd,
            "rng": _rng_state(),
            "extra": dict(extra or {}),
            "meta": meta,
        }

    def _write(self, payload, final):
        tmp = final + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, final)        # atomic: a crash mid-save never corrupts the last good file
        self.stats["last_path"] = final
        self.stats["last_bytes"] = os.path.getsize(final)
        self._cleanup()

    def _write_in_background(self, payload, final):
        start = time.perf_counter()
        try:
            self._write(payload, final)
        except BaseException as e:                  # kept and raised in the main thread
            self._error = e
            try:
                os.remove(final + ".tmp")
            except OSError:
                pass
        finally:
            self.stats["background_s"] += time.perf_counter() - start

    def wait(self):
        """Wait for a background save to finish; raise if it failed."""
        if self._thread is not None:
            if self._thread.is_alive():
                start = time.perf_counter()
                self._thread.join()
                self.stats["waited_s"] += time.perf_counter() - start
            self._thread = None
        self.check()

    def check(self):
        """Raise the error of a finished background save, if there was one. Doesn't wait."""
        if self._error is not None and (self._thread is None or not self._thread.is_alive()):
            error, self._error = self._error, None
            raise RuntimeError(f"Saving a checkpoint in the background failed: {error}") from error

    @property
    def pending(self):
        """True while a background save is still writing."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def writer(self):
        """The thread writing a background save, or None."""
        return self._thread

    # ---------- loading ----------
    def latest(self):
        """Path of the newest checkpoint in the folder, or None."""
        files = sorted(glob.glob(os.path.join(self.directory, "ckpt_step*.pt")))
        return files[-1] if files else None

    def load(self, model, optimizer, scheduler=None, scaler=None, path=None, map_location="cpu",
             fingerprint=None):
        """Restore everything. Returns the saved train_state dict, or None if nothing to
        resume. Reads checkpoints in any mode (whatever this manager's own mode is), those
        written by Efficio 0.1.x, and early 0.2.0 "light" checkpoints (as "smallest").
        fingerprint: raise DifferentRunError, before changing anything, if the checkpoint has
        a different run fingerprint (checkpoints without one are accepted)."""
        self.wait()
        path = path or self.latest()
        if path is None:
            return None
        # Our own file holds RNG states and plain dicts, so the weights_only loader can't read it.
        payload = torch.load(path, map_location=map_location, weights_only=False)
        meta = dict(payload.get("meta") or {})              # 0.1.x checkpoints have none
        saved_fp = meta.get("fingerprint")
        if fingerprint and saved_fp and saved_fp.get("hash") != fingerprint["hash"]:
            raise DifferentRunError(f"{path} (step {payload['train_state']['step']})",
                                    saved_fp, fingerprint)
        mode = meta.get("mode", payload.get("mode", "exact"))
        if meta.get("format_version") == 2 and mode == "light":
            mode = meta["mode"] = "smallest"                # early builds' "light"
            meta["written_as"] = "light (format 2)"
        model_sd = payload["model"]
        opt_sd = payload["optimizer"]
        if meta.get("model_dtypes"):                        # fp16 weights: cast back
            model_sd = light.decode_model(model_sd, meta["model_dtypes"])
        if mode in ("light", "smallest"):
            opt_sd = light.decode_optimizer(opt_sd)
        if meta.get("trainable_only"):
            self._load_trainable_only(model, model_sd, meta.get("frozen", {}), path)
        else:
            model.load_state_dict(model_sd)
        optimizer.load_state_dict(opt_sd)
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        _set_rng_state(payload["rng"])
        self.extra = payload.get("extra") or {}     # older checkpoints have none
        self.meta = meta
        return payload["train_state"]

    @staticmethod
    def _load_trainable_only(model, model_sd, frozen, path):
        current = model.state_dict()
        problems = []
        for name, saved in frozen.items():
            if name not in current:
                problems.append(f"{name}: not in the model")
            elif not _same_signature(saved, tensor_signature(current[name])):
                problems.append(f"{name}: different weights, shape or dtype")
        if problems:
            shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more"
                                               if len(problems) > 5 else "")
            raise CheckpointMismatchError(
                f"{path} was saved with trainable parameters only, and the frozen parameters "
                f"of this model don't match the ones it was trained with ({shown}). Load the "
                "same pretrained weights and freeze the same layers before run.prepare(), or "
                "start a new run with a different checkpoint_dir.")
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        not_frozen = [k for k in missing if k not in frozen]
        if not_frozen or unexpected:
            raise CheckpointMismatchError(
                f"{path} doesn't fit this model: missing {not_frozen[:5]}, "
                f"unexpected {list(unexpected)[:5]}.")

    def _cleanup(self):
        files = sorted(glob.glob(os.path.join(self.directory, "ckpt_step*.pt")))
        for old in files[:-self.keep_last]:
            os.remove(old)
