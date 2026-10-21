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
import itertools
import signal
import time

import torch

from .checkpoint import CheckpointManager
from .env import default_checkpoint_dir, describe, detect_hardware
from .interfaces import Budget, TrainState
from .report.profiler import GpuUtilSampler, Profiler, tips
from .speed.dataloader import optimize_loader
from .speed.gpu_data import GPULoader, fits_on_gpu
from .speed.memory_format import use_channels_last

_END = object()


def _move(obj, device, channels_last=False):
    if torch.is_tensor(obj):
        if channels_last and obj.dim() == 4 and obj.is_floating_point():
            return obj.to(device, non_blocking=True, memory_format=torch.channels_last)
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move(v, device, channels_last) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(v, device, channels_last) for v in obj)
    return obj


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # PyTorch 2.3+
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older PyTorch


class Run:
    def __init__(self, budget=None, checkpoint_dir=None, save_every_minutes=15,
                 grad_accum_steps=1, resume=True, verbose=True,
                 optimize_data=True, channels_last="auto"):
        self.profiler = Profiler()               # setup time is measured from here
        self.hw = detect_hardware()
        self.device = torch.device(self.hw.device)
        self.budget = Budget.parse(budget)
        self.state = TrainState(grad_accum_steps=grad_accum_steps)
        self.ckpt = CheckpointManager(checkpoint_dir or default_checkpoint_dir(self.hw.platform))
        self.save_every = save_every_minutes * 60
        self.resume = resume
        self.verbose = verbose
        self.optimize_data = optimize_data
        self.channels_last_setting = channels_last
        self.channels_last = False
        self.optimizations = [self.hw.precision]  # what Efficio applied, for the report
        self._gpu = GpuUtilSampler(interval=1.0)
        self._loader_logged = False
        self._gpu_loader_fits = False    # report tip: the dataset could live on the GPU

        self.model = self.optimizer = self.scheduler = None
        self.scaler = _make_scaler(self.hw.precision == "fp16")
        self._micro = 0                  # micro-batches since the last optimizer step
        self._last_save = time.time()
        self._stop_reason = None
        self._signal_received = False
        self._saves = 0
        self._resumed_from = None
        self._log(describe(self.hw))
        if self.hw.device == "cuda":
            torch.backends.cudnn.benchmark = True
            self.optimizations.append("cudnn.benchmark")
            self._log("[efficio] cudnn.benchmark on")

    # ---------- setup ----------
    def prepare(self, model, optimizer, scheduler=None):
        self.model = model.to(self.device)
        self.optimizer, self.scheduler = optimizer, scheduler
        use, reason = use_channels_last(self.channels_last_setting, self.hw.device, self.model)
        if use:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.channels_last = True
            self.optimizations.append("channels_last")
            self._log("[efficio] channels_last on (conv model on GPU)")
        elif self.channels_last_setting is True:
            self._log(f"[efficio] channels_last skipped: {reason}")
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
        loader = self._prepare_loader(loader)
        prof = self.profiler
        if self._stop_reason == "finished":     # an earlier loop() ended normally; train again
            self._stop_reason = None
        old_handler = self._install_signal_handler()
        try:
            for epoch in range(self.state.epoch, epochs):
                self.state.epoch = epoch
                skip = start = self.state.batch_in_epoch
                if hasattr(loader, "set_epoch"):
                    loader.set_epoch(epoch)
                if skip and hasattr(loader, "set_start_batch"):
                    loader.set_start_batch(skip)    # GPULoader: jump there without loading
                else:
                    start = 0
                batches = iter(loader)
                for i in itertools.count(start):
                    wait_start = time.perf_counter()
                    batch = next(batches, _END)
                    if batch is _END:
                        break
                    if i < skip:      # already trained before the resume
                        continue      # (DataLoader: data is still loaded, then dropped)
                    fetched = time.perf_counter()
                    if self.state.batch_size == 0 and getattr(loader, "batch_size", None):
                        self.state.batch_size = loader.batch_size
                    self.state.batch_in_epoch = i + 1
                    # Time up to the first trained batch (worker start-up, resume skipping) is setup
                    first = prof.first_batch_at is None
                    if first:
                        prof.first_batch(fetched)
                        # Start sampling only now: DataLoader workers are already forked
                        # (forking with a running thread can deadlock) and setup is excluded
                        if self.hw.device == "cuda":
                            self._gpu.start()
                    yield _move(batch, self.device, self.channels_last)
                    prof.add_step(0.0 if first else fetched - wait_start,
                                  time.perf_counter() - fetched)
                    if self._stop_reason:
                        return
                self.state.epoch, self.state.batch_in_epoch = epoch + 1, 0
            self._stop_reason = self._stop_reason or "finished"
            self.save()
        finally:
            self._gpu.stop()
            prof.finish()
            self._restore_signal_handler(old_handler)

    def _prepare_loader(self, loader):
        if isinstance(loader, GPULoader):
            self._note_optimization("GPULoader (data on the GPU)")
            return loader
        if self.hw.device == "cuda":
            self._gpu_loader_fits = fits_on_gpu(getattr(loader, "dataset", None), self.device)
        if not self.optimize_data:
            return loader
        loader, changes = optimize_loader(loader, self.hw)
        applied = [c for c in changes if not c.startswith(("unchanged", "already", "note"))]
        if applied:
            self._note_optimization(f"dataloader ({', '.join(applied)})")
        if not self._loader_logged:
            self._log(f"[efficio] dataloader: {', '.join(changes)}")
            self._loader_logged = True
        return loader

    def _note_optimization(self, text):
        if text not in self.optimizations:      # loop() may be called more than once
            self.optimizations.append(text)

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
        """Catch SIGTERM while training. Returns the previous handler, or None if ours
        could not be installed."""
        def handler(signum, frame):
            self._signal_received = True       # save happens at the next optimizer step
        try:
            return signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):           # not in the main thread, or unsupported
            return None

    @staticmethod
    def _restore_signal_handler(old):
        # Put the previous handler back so that, after training, SIGTERM stops the process
        # (or notebook kernel) as usual instead of only setting a flag nobody reads.
        if old is None:
            return
        try:
            signal.signal(signal.SIGTERM, old)
        except (ValueError, OSError, TypeError):
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
        prof = self.profiler
        util = self._gpu.average
        lines += [
            f"  setup / training: {prof.setup_s / 60:.1f} min / {prof.train_s / 60:.1f} min "
            f"(setup {prof.setup_pct:.0f}% of total)",
            f"  data-wait       : {prof.data_wait_pct:.0f}% of training time",
            f"  GPU utilization : {f'{util:.0f}% average' if util is not None else 'not measured'}",
        ]
        if self.hw.device == "cuda":
            lines.append(f"  GPU session used: {mins:.1f} min")
        lines.append(f"  optimizations   : {', '.join(self.optimizations)}")
        advice = tips(prof.data_wait_pct, util, prof.setup_pct,
                      on_gpu=self.hw.device == "cuda",
                      gpu_loader_fits=self._gpu_loader_fits) if prof.steps else []
        if advice:
            lines.append("  tips:")
            lines += ["    - " + t.replace("\n", "\n        ") for t in advice]
        print("\n".join(lines))

    def _log(self, msg):
        if self.verbose:
            print(msg)
