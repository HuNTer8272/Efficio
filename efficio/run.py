"""The Run class: the one object users interact with.

Usage:
    run = efficio.Run(budget="2h")
    model, optimizer = run.prepare(model, optimizer)
    for batch in run.loop(loader, epochs=20):
        with run.autocast():
            loss = compute_loss(model, batch)
        run.backward(loss)
        run.clip_grad_norm(1.0)      # optional, works with fp16 and gradient accumulation
        run.step()
    run.report()
"""

import contextlib
import itertools
import os
import signal
import time

import torch

from .budget import shutdown
from .checkpoint import CheckpointManager
from .checkpoint.manager import (
    DifferentRunError,
    StorageInUseError,
    _efficio_version,
    _rng_state,
    _set_rng_state,
    checkpoint_step,
    run_fingerprint,
)
from .data.image_cache import CachedImageDataset
from .env import default_checkpoint_dir, describe, detect_hardware
from .interfaces import Budget, TrainState
from .report.profiler import GpuUtilSampler, Profiler, tips
from .speed.dataloader import optimize_loader, seeded_shuffle
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


def _decodes_image_files(dataset):
    """True for datasets that read and decode image files on every access, such as
    torchvision ImageFolder or a custom dataset with a list of image paths (report tip)."""
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if dataset is None or isinstance(dataset, CachedImageDataset):
        return False
    if any(hasattr(dataset, a) for a in ("samples", "imgs", "image_paths", "img_paths")):
        return True
    return "image" in type(dataset).__name__.lower()


def _size(nbytes):
    for unit, factor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if nbytes >= factor:
            return f"{nbytes / factor:.1f} {unit}"
    return f"{nbytes} B"


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # PyTorch 2.3+
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older PyTorch


class Run:
    """One training run: device and precision choice, data loading, checkpoints, the time
    budget and the end-of-run report.

    budget: e.g. "2h" or "90m"; the run saves and stops 10 minutes before it ends.
    checkpoint_dir: where checkpoints go (default: a sensible folder per platform).
    save_every_minutes: periodic checkpoint interval.
    grad_accum_steps: micro-batches per optimizer step.
    resume: "auto" (default) or True: continue from the latest checkpoint (in checkpoint_dir,
        or in storage if that one is newer). A checkpoint saved by a different run (other
        parameter names or shapes, or another optimizer type) is never resumed: with "auto"
        a warning is printed and training starts fresh, with True it raises
        CheckpointMismatchError. False: always start fresh.
    optimize_data: tune DataLoaders passed to loop() (workers, pinned memory, prefetching).
    channels_last: "auto", True or False; channels_last memory format for conv models on GPU.
    checkpoint_mode: how checkpoints are stored; any mode can resume from any other.
        "exact" (default, safest): everything at full precision, bit-identical resume.
        "light": weights at full precision, optimizer state in 8 bits; about 2x smaller for
        Adam, the weights resume exactly and the effect on training is tiny.
        "smallest": light plus fp16 weights; about 3x smaller, a slight effect after resume.
    save_trainable_only: "auto" (default: only when some parameters are frozen), True or
        False. Frozen parameters are left out of checkpoints; resuming checks that the
        model's frozen weights are the same ones.
    async_save: write periodic checkpoints in a background thread (default True). Budget,
        shutdown-signal and final saves always finish before training stops.
    storage: also keep checkpoints outside this machine, so the run survives the end of the
        session and can resume on another platform: "hf://user/repo" (a private Hugging Face
        repo, needs efficio[hub]), a folder path such as a mounted Google Drive folder, or an
        efficio.storage.Storage object. Each checkpoint is uploaded in the background after
        it is saved; budget, shutdown-signal and final saves wait for the upload. When there
        is no newer checkpoint on this machine, prepare() downloads the latest one from there.
    keep_remote: how many checkpoints to keep in storage (default 3).
    overwrite_storage: if storage already holds checkpoints of a different run (other
        parameter names or shapes, or another optimizer type), prepare() refuses to start
        unless this is True (default False). With True, that run's checkpoints are never
        resumed and are replaced by this run's as it saves.
    upload_timeout: seconds a budget, shutdown-signal or final save waits for its upload.
    shutdown_when_done: on Colab, release the runtime after run.report() once the final
        checkpoint is confirmed in storage, so no GPU quota is wasted (default False).
    """

    def __init__(self, budget=None, checkpoint_dir=None, save_every_minutes=15,
                 grad_accum_steps=1, resume="auto", verbose=True,
                 optimize_data=True, channels_last="auto", checkpoint_mode="exact",
                 save_trainable_only="auto", async_save=True, storage=None, keep_remote=3,
                 overwrite_storage=False, upload_timeout=120, shutdown_when_done=False):
        if save_trainable_only not in ("auto", True, False):
            raise ValueError('save_trainable_only must be "auto", True or False, '
                             f"not {save_trainable_only!r}")
        if resume not in ("auto", True, False):
            raise ValueError(f'resume must be "auto", True or False, not {resume!r}')
        self.profiler = Profiler()               # setup time is measured from here
        self.hw = detect_hardware()
        self.device = torch.device(self.hw.device)
        self.budget = Budget.parse(budget)
        self.state = TrainState(grad_accum_steps=grad_accum_steps)
        self.verbose = verbose
        self.ckpt = CheckpointManager(
            checkpoint_dir or default_checkpoint_dir(self.hw.platform,
                                                     session_disk=storage is not None),
            mode=checkpoint_mode, async_save=async_save)
        self.remote = self._open_storage(storage, keep_remote) if storage is not None else None
        self.overwrite_storage = overwrite_storage
        self.upload_timeout = upload_timeout
        self._final_upload = None                # (path, status) of the last waited-for upload
        self.shutdown_when_done = shutdown_when_done
        self.save_trainable_only = save_trainable_only
        self.save_every = save_every_minutes * 60
        self.resume = resume
        self.optimize_data = optimize_data
        self.channels_last_setting = channels_last
        self.channels_last = False
        self.optimizations = [self.hw.precision]  # what Efficio applied, for the report
        self._gpu = GpuUtilSampler(interval=1.0)
        self._loader_logged = False
        self._gpu_loader_fits = False    # report tip: the dataset could live on the GPU
        self._image_files = False        # report tip: images decoded from files every epoch
        self._shuffle_seed = None        # base seed of a shuffling DataLoader, saved for resume
        self._set_epoch = None

        self.model = self.optimizer = self.scheduler = None
        self._frozen = 0                 # frozen parameter tensors, counted in prepare()
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
        if self.remote is not None:
            self._log(f"[efficio] storage: {self.remote.storage.describe()} (keeps the last "
                      f"{self.remote.keep}); each checkpoint is uploaded in the background")
        if shutdown_when_done:
            self._log(shutdown.startup_notice(self.hw.platform, self.remote is not None))

    def _open_storage(self, spec, keep):
        from .storage import LocalFolder, RemoteCheckpoints, open_storage
        backend = open_storage(spec)
        if (isinstance(backend, LocalFolder) and
                os.path.abspath(backend.path) == os.path.abspath(self.ckpt.directory)):
            raise ValueError(f"storage ({backend.path}) must be a different folder from "
                             "checkpoint_dir; it holds copies of the checkpoints there")
        backend.connect(self._log)
        return RemoteCheckpoints(backend, keep=keep, platform=self.hw.platform)

    # ---------- setup ----------
    def prepare(self, model, optimizer, scheduler=None):
        """Move the model to the device, apply channels_last if chosen and resume from the
        latest checkpoint. Returns (model, optimizer)."""
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
        frozen = self._frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        self.ckpt.trainable_only = (frozen > 0 if self.save_trainable_only == "auto"
                                    else bool(self.save_trainable_only))
        if self.ckpt.mode == "light":
            self._log("[efficio] checkpoints: light mode (full-precision weights, "
                      "8-bit optimizer state)")
        elif self.ckpt.mode == "smallest":
            self._log("[efficio] checkpoints: smallest mode (fp16 weights, "
                      "8-bit optimizer state)")
        if self.ckpt.trainable_only and frozen:
            self._log(f"[efficio] checkpoints: trainable parameters only "
                      f"({frozen} frozen tensors are checked, not saved)")
        self.ckpt.fingerprint = run_fingerprint(self.model, optimizer)
        if self.remote is not None:
            self._check_storage_owner()
        if self.resume:
            path = self._resume_file()
            try:
                saved = self.ckpt.load(self.model, optimizer, scheduler, self.scaler, path=path,
                                       map_location="cpu", fingerprint=self.ckpt.fingerprint)
            except DifferentRunError as e:
                self._different_run(e)
                saved = self._move_aside_local_checkpoints()
            if saved is not None:
                for k, v in saved.items():
                    setattr(self.state, k, v)
                self._resumed_from = (saved["epoch"], saved["batch_in_epoch"])
                self._shuffle_seed = self.ckpt.extra.get("shuffle_seed")
                self._log(f"[efficio] resumed from epoch {saved['epoch']}, "
                          f"batch {saved['batch_in_epoch']}, step {saved['step']}")
        return self.model, self.optimizer

    def _resume_file(self):
        """The local checkpoint, or the latest one in storage (downloaded) if that is newer,
        e.g. in a new session or on another platform."""
        local = self.ckpt.latest()
        if self.remote is None:
            return local
        from .storage import StorageError
        where = self.remote.storage.describe()
        try:
            entry = self.remote.latest()
        except StorageError as e:
            if local is None:
                raise StorageError(f"Could not read the checkpoint list in {where}: {e}. "
                                   "Nothing was changed there. Check the connection and "
                                   "token, or pass resume=False to start fresh.") from None
            self._log(f"[efficio] could not read {where} ({e}); resuming from this machine")
            return local
        if entry is None or (local is not None and checkpoint_step(local) >= entry["step"]):
            return local
        if self._other_run(entry):      # only with overwrite_storage=True; never resumed
            return local
        if local is not None:
            self._log(f"[efficio] {where} has a newer checkpoint (step {entry['step']}) than "
                      f"this machine (step {checkpoint_step(local)})")
        path = self.remote.download(entry, self.ckpt.directory)
        self._log(f"[efficio] downloaded {entry['file']} from {where}: step {entry['step']}, "
                  f"{entry.get('mode', 'exact')} mode, saved on {entry.get('platform', '?')} "
                  f"at {entry.get('time', '?')}; sha256 verified")
        return path

    def _other_run(self, entry):
        """True for a manifest entry saved by a different run (entries from before the
        fingerprint have none and count as this run's)."""
        saved = entry.get("fingerprint")
        return bool(saved) and saved.get("hash") != self.ckpt.fingerprint["hash"]

    def _check_storage_owner(self):
        """Refuse to write into storage that holds another run's checkpoints, unless
        overwrite_storage=True. Runs before anything is written there."""
        from .storage import StorageError
        where = self.remote.storage.describe()
        try:
            others = [e for e in self.remote.entries() if self._other_run(e)]
        except StorageError as e:
            raise StorageError(f"Could not read the checkpoint list in {where}: {e}. Nothing "
                               "was changed there; check the connection and token.") from None
        if not others:
            return
        newest = others[-1]
        if not self.overwrite_storage:
            raise StorageInUseError(f"{where} (step {newest['step']})", newest["fingerprint"],
                                    self.ckpt.fingerprint)
        self.remote.replace_fingerprints = {e["fingerprint"]["hash"] for e in others}
        print(f"[efficio] WARNING: overwrite_storage=True: {where} holds {len(others)} "
              f"checkpoint(s) of a different run (newest at step {newest['step']}). They are "
              "not resumed, and are deleted once this run's first checkpoint is uploaded.")

    def _different_run(self, error):
        """A checkpoint from another run: raise with resume=True, else warn and go on."""
        if self.resume is True:
            raise error
        print(f"[efficio] WARNING: not resuming from {error.where}: it was saved by a different "
              f"run ({error.saved.get('optimizer')}, {error.saved.get('parameters')} parameter "
              f"tensors; this run: {error.current['optimizer']}, "
              f"{error.current['parameters']}). Starting fresh. New checkpoints go to the same "
              "places and replace the old ones there; use another checkpoint_dir or storage "
              "to keep them, or pass resume=True to make this an error.")

    def _move_aside_local_checkpoints(self):
        """Starting fresh next to another run's checkpoints: move them into a subfolder, so
        they are neither resumed nor mixed up with this run's (their steps may be higher).
        Returns None (nothing resumed)."""
        files = [f for f in os.listdir(self.ckpt.directory)
                 if f.startswith("ckpt_step") and f.endswith(".pt")]
        aside = os.path.join(self.ckpt.directory,
                             time.strftime("other_run_%Y%m%d_%H%M%S"))
        os.makedirs(aside, exist_ok=True)
        for f in files:
            os.replace(os.path.join(self.ckpt.directory, f), os.path.join(aside, f))
        print(f"[efficio] moved {len(files)} checkpoint file(s) of the other run to {aside}")
        return None

    @contextlib.contextmanager
    def autocast(self):
        """Mixed precision context for the forward pass: fp16 or bf16 on GPU, no-op on CPU."""
        if self.hw.precision == "fp32":
            yield
            return
        dtype = torch.bfloat16 if self.hw.precision == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    @contextlib.contextmanager
    def evaluate_mode(self):
        """For validation: model.eval(), no gradients and the run's mixed precision. The
        model's previous train/eval mode is restored afterwards."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.autocast():
                yield
        finally:
            self.model.train(was_training)

    # ---------- training loop ----------
    def loop(self, loader, epochs, gpu_transform=None):
        """Yields batches already on the right device. Skips work done before a resume
        and stops cleanly (with a checkpoint) when the time budget runs low.

        gpu_transform: optional function applied to each batch after it is on the device
        (without gradients), e.g. efficio.on_inputs(augment) for GPU augmentation. Its
        randomness comes from the global torch RNG, which checkpoints save."""
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
                if self._set_epoch is not None:
                    self._set_epoch(epoch)          # same shuffle order as before a resume
                rng_before_skip = None
                if skip and hasattr(loader, "set_start_batch"):
                    loader.set_start_batch(skip)    # GPULoader: jump there without loading
                else:
                    start = 0
                    if skip:
                        # Loading the skipped batches may draw random augmentations; put the
                        # RNG back afterwards so the next batch continues the saved stream.
                        rng_before_skip = _rng_state()
                batches = iter(loader)
                for i in itertools.count(start):
                    if rng_before_skip is not None and i == skip:
                        _set_rng_state(rng_before_skip)
                        rng_before_skip = None
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
                    batch = _move(batch, self.device, self.channels_last and not gpu_transform)
                    if gpu_transform is not None:
                        with torch.no_grad():
                            batch = gpu_transform(batch)
                        if self.channels_last:      # the transform makes new float images
                            batch = _move(batch, self.device, True)
                    yield batch
                    prof.add_step(0.0 if first else fetched - wait_start,
                                  time.perf_counter() - fetched)
                    if self._stop_reason:
                        return
                if rng_before_skip is not None:            # epoch shorter than the skip
                    _set_rng_state(rng_before_skip)
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
        self._image_files = _decodes_image_files(getattr(loader, "dataset", None))
        if self.optimize_data:
            loader, changes = optimize_loader(loader, self.hw)
            applied = [c for c in changes if not c.startswith(("unchanged", "already", "note"))]
            if applied:
                self._note_optimization(f"dataloader ({', '.join(applied)})")
            if not self._loader_logged:
                self._log(f"[efficio] dataloader: {', '.join(changes)}")
                self._loader_logged = True
        loader, self._set_epoch = seeded_shuffle(loader, self._shuffle_seed)
        if self._set_epoch is not None:
            self._shuffle_seed = self._set_epoch.base_seed
        return loader

    def _note_optimization(self, text):
        if text not in self.optimizations:      # loop() may be called more than once
            self.optimizations.append(text)

    def backward(self, loss):
        """Backward pass with gradient scaling (fp16) and gradient accumulation."""
        self.scaler.scale(loss / self.state.grad_accum_steps).backward()

    def clip_grad_norm(self, max_norm, norm_type=2.0):
        """Clip the gradient norm of the optimizer's parameters. Call it between backward()
        and step(). With fp16 the gradients are unscaled first, so max_norm is in real units.
        With gradient accumulation it only acts on the micro-batch where step() will update
        the weights. Returns the total norm before clipping, or None when it did nothing."""
        if self._micro + 1 < self.state.grad_accum_steps:
            return None                 # gradients are still accumulating
        self.scaler.unscale_(self.optimizer)        # no-op when the scaler is disabled
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    def step(self):
        """Optimizer and scheduler step once enough micro-batches have accumulated; also
        saves periodically and stops when the time budget runs low."""
        self._micro += 1
        if self._micro < self.state.grad_accum_steps:
            return
        self._micro = 0
        self.ckpt.check()               # raise a failed background save here, not later
        if self.remote is not None:
            self.remote.check()         # and a failed upload
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.step += 1
        self._maybe_save_or_stop()

    # ---------- checkpointing and budget ----------
    def save(self, blocking=True):
        """Save a checkpoint now and return its path. By default it waits until the file is
        written; blocking=False writes it in the background when async_save is on. With
        storage set, the file is then uploaded; a blocking save also waits for the upload
        (up to upload_timeout seconds) and says whether it succeeded."""
        extra = {"shuffle_seed": self._shuffle_seed} if self._shuffle_seed is not None else None
        if self.remote is not None:
            self.remote.wait()          # one upload at a time; its file must stay until done
        path = self.ckpt.save(self.state, self.model, self.optimizer, self.scheduler, self.scaler,
                              extra=extra, blocking=blocking)
        self._last_save = time.time()
        self._saves += 1
        if self.ckpt.pending:
            self._log(f"[efficio] checkpoint saving in the background: {path}")
        else:
            self._log(f"[efficio] checkpoint saved: {path}")
        if self.remote is not None:
            self._upload(path, blocking)
        return path

    def _upload(self, path, blocking):
        info = {"step": self.state.step, "epoch": self.state.epoch,
                "batch_in_epoch": self.state.batch_in_epoch, "mode": self.ckpt.mode,
                "trainable_only": bool(self.ckpt.trainable_only and self._frozen),
                "fingerprint": self.ckpt.fingerprint, "efficio_version": _efficio_version()}
        if not blocking:
            self.remote.upload(path, info, after=self.ckpt.writer)  # raises an earlier failure
            return
        status, message = self.remote.finish()      # an earlier upload: report, don't raise
        if status == "failed":
            print(f"[efficio] an earlier upload failed: {message}")
        self.remote.upload(path, info)
        status, message = self.remote.finish(self.upload_timeout)
        self._final_upload = (path, status)
        if status == "ok":
            self._log(f"[efficio] upload complete: {message}")
        elif status == "failed":
            print(f"[efficio] UPLOAD FAILED: {message}\n[efficio] the checkpoint is only on "
                  f"this machine: {path}")
        else:
            print(f"[efficio] upload NOT confirmed: {message}")

    def _maybe_save_or_stop(self):
        if self._signal_received:
            self._stop_reason = "shutdown signal"
        elif self.budget.nearly_out():
            self._stop_reason = "time budget nearly used"
        if self._stop_reason:
            self._log(f"[efficio] stopping: {self._stop_reason}")
            self.save()                         # waits for any background save, then writes
        elif time.time() - self._last_save >= self.save_every:
            self.save(blocking=False)

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
        """True when the run stopped early (time budget or shutdown signal)."""
        return self._stop_reason is not None and self._stop_reason != "finished"

    # ---------- reporting ----------
    def report(self):
        """Print the end-of-run report: progress, time split, GPU use and tips."""
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
        lines += self._checkpoint_report()
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
                      gpu_loader_fits=self._gpu_loader_fits,
                      image_files=self._image_files,
                      train_s=prof.train_s) if prof.steps else []
        if advice:
            lines.append("  tips:")
            lines += ["    - " + t.replace("\n", "\n        ") for t in advice]
        print("\n".join(lines))
        self._maybe_shut_down()

    def _maybe_shut_down(self):
        """shutdown_when_done: release the Colab runtime, only once training ended and the
        final checkpoint is confirmed in storage."""
        if not self.shutdown_when_done:
            return
        if self.hw.platform != "colab":
            self._log(shutdown.startup_notice(self.hw.platform, self.remote is not None))
            return
        if self._stop_reason is None:
            print("[efficio] not shutting down: run.loop() has not ended")
            return
        path, status = self._final_upload or (None, None)
        if self.remote is None or status != "ok" or path != self.ckpt.stats["last_path"]:
            print("[efficio] NOT shutting down: the final checkpoint is not confirmed in "
                  "storage, and it would be lost with the runtime")
            return
        shutdown.release_colab_runtime(self._log)

    def _checkpoint_report(self):
        try:
            self.ckpt.wait()                    # finish a background save before reporting
            failed = ""
        except RuntimeError as e:
            failed = f" (last background save FAILED: {e.__cause__ or e})"
        st = self.ckpt.stats
        kind = self.ckpt.mode + (", trainable parameters only" if self.ckpt.trainable_only else "")
        size = (f", last file {_size(st['last_bytes'])}" if st["last_bytes"] is not None
                else "")
        lines = [f"  checkpoint mode : {kind}{size}{failed}"]
        if st["saves"]:
            hidden = max(st["background_s"] - st["waited_s"], 0.0)
            line = f"  save time       : {st['blocking_s']:.1f} s blocking training"
            if st["background_s"]:
                line += (f", {st['background_s']:.1f} s written in the background "
                         f"({hidden:.1f} s hidden)")
            lines.append(line)
        return lines + self._storage_report()

    def _storage_report(self):
        if self.remote is None:
            return []
        status, message = self.remote.finish(self.upload_timeout)   # a periodic upload
        st = self.remote.stats
        lines = [f"  storage         : {self.remote.storage.describe()} "
                 f"(keeps the last {self.remote.keep})"]
        if st["downloaded"]:
            d = st["downloaded"]
            lines.append(f"  downloaded      : {d['file']} (step {d['step']}, saved on "
                         f"{d.get('platform', '?')}), sha256 verified")
        if status == "failed" or st["last_ok"] is False:
            why = message if status == "failed" else st["last_error"]
            lines.append(f"  last upload     : FAILED ({why})")
        elif status == "running":
            lines.append(f"  last upload     : still running ({message})")
        if st["last_file"]:
            lines.append(f"  last uploaded   : {st['last_file']}, {_size(st['last_bytes'])} at "
                         f"{st['last_time']}")
        if st["uploads"]:
            hidden = max(st["upload_s"] - st["waited_s"], 0.0)
            lines.append(f"  upload time     : {st['upload_s']:.1f} s in the background "
                         f"({hidden:.1f} s hidden)")
        return lines

    def _log(self, msg):
        if self.verbose:
            print(msg)
