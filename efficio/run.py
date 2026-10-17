"""The Run class: the one object users interact with.

Usage:
    run = efficio.Run(budget="2h")
    model, optimizer = run.prepare(model, optimizer)
    for batch in run.loop(loader, epochs=20):
        with run.autocast():
            loss = compute_loss(model, batch)
        run.backward(loss)
        run.step()
    run.report()
"""

import contextlib
import signal
import time

import torch

from .checkpoint import CheckpointManager
from .env import default_checkpoint_dir, describe, detect_hardware
from .interfaces import Budget, TrainState


def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(v, device) for v in obj)
    return obj


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # PyTorch 2.3+
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older PyTorch


class Run:
    def __init__(self, budget=None, checkpoint_dir=None, save_every_minutes=15,
                 grad_accum_steps=1, resume=True, verbose=True):
        self.hw = detect_hardware()
        self.device = torch.device(self.hw.device)
        self.budget = Budget.parse(budget)
        self.state = TrainState(grad_accum_steps=grad_accum_steps)
        self.ckpt = CheckpointManager(checkpoint_dir or default_checkpoint_dir(self.hw.platform))
        self.save_every = save_every_minutes * 60
        self.resume = resume
        self.verbose = verbose

        self.model = self.optimizer = self.scheduler = None
        self.scaler = _make_scaler(self.hw.precision == "fp16")
        self._micro = 0                  # micro-batches since the last optimizer step
        self._last_save = time.time()
        self._stop_reason = None
        self._signal_received = False
        self._saves = 0
        self._resumed_from = None
        self._install_signal_handler()
        self._log(describe(self.hw))

    # ---------- setup ----------
    def prepare(self, model, optimizer, scheduler=None):
        self.model = model.to(self.device)
        self.optimizer, self.scheduler = optimizer, scheduler
        if self.resume:
            saved = self.ckpt.load(self.model, optimizer, scheduler, self.scaler,
                                   map_location="cpu")
            if saved is not None:
                for k, v in saved.items():
                    setattr(self.state, k, v)
                self._resumed_from = (saved["epoch"], saved["batch_in_epoch"])
                self._log(f"[efficio] resumed from epoch {saved['epoch']}, "
                          f"batch {saved['batch_in_epoch']}, step {saved['step']}")
        return self.model, self.optimizer

    @contextlib.contextmanager
    def autocast(self):
        if self.hw.precision == "fp32":
            yield
            return
        dtype = torch.bfloat16 if self.hw.precision == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    # ---------- training loop ----------
    def loop(self, loader, epochs):
        """Yields batches already on the right device. Skips work done before a resume
        and stops cleanly (with a checkpoint) when the time budget runs low."""
        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            skip = self.state.batch_in_epoch
            for i, batch in enumerate(loader):
                if i < skip:          # already trained before the resume
                    continue          # (Phase 1: data is still loaded; faster skipping comes later)
                if self.state.batch_size == 0 and hasattr(loader, "batch_size") and loader.batch_size:
                    self.state.batch_size = loader.batch_size
                self.state.batch_in_epoch = i + 1
                yield _move(batch, self.device)
                if self._stop_reason:
                    return
            self.state.epoch, self.state.batch_in_epoch = epoch + 1, 0
        self._stop_reason = self._stop_reason or "finished"
        self.save()

    def backward(self, loss):
        self.scaler.scale(loss / self.state.grad_accum_steps).backward()

    def step(self):
        self._micro += 1
        if self._micro < self.state.grad_accum_steps:
            return
        self._micro = 0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.step += 1
        self._maybe_save_or_stop()

    # ---------- checkpointing and budget ----------
    def save(self):
        path = self.ckpt.save(self.state, self.model, self.optimizer, self.scheduler, self.scaler)
        self._last_save = time.time()
        self._saves += 1
        self._log(f"[efficio] checkpoint saved: {path}")
        return path

    def _maybe_save_or_stop(self):
        if self._signal_received:
            self._stop_reason = "shutdown signal"
        elif self.budget.nearly_out():
            self._stop_reason = "time budget nearly used"
        if self._stop_reason:
            self._log(f"[efficio] stopping: {self._stop_reason}")
            self.save()
        elif time.time() - self._last_save >= self.save_every:
            self.save()

    def _install_signal_handler(self):
        def handler(signum, frame):
            self._signal_received = True       # save happens at the next optimizer step
        try:
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):           # not in the main thread, or unsupported
            pass

    @property
    def should_stop(self):
        return self._stop_reason is not None and self._stop_reason != "finished"

    # ---------- reporting ----------
    def report(self):
        mins = self.budget.elapsed() / 60
        lines = [
            "[efficio] ---- run report ----",
            f"  platform/device : {self.hw.platform} / {self.hw.device} ({self.hw.precision})",
            f"  resumed from    : {self._resumed_from or 'fresh start'}",
            f"  reached         : epoch {self.state.epoch}, step {self.state.step}",
            f"  time this run   : {mins:.1f} min",
            f"  checkpoints     : {self._saves} saved to {self.ckpt.directory}",
            f"  ended because   : {self._stop_reason or 'loop not finished'}",
        ]
        print("\n".join(lines))

    def _log(self, msg):
        if self.verbose:
            print(msg)
