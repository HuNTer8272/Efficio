# Changelog

All notable changes to Efficio are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Light checkpoints: `Run(checkpoint_mode="light")` stores floating-point weights in fp16 (cast back to their original dtype on load) and large optimizer state tensors with 8-bit blockwise quantization (pure PyTorch). Files are about 3x smaller for Adam-trained models. Scheduler, gradient scaler, RNG states and position are saved exactly; the resume is close but not bit-identical. The default stays `"exact"`.
- Trainable-only checkpoints: `Run(save_trainable_only="auto" | True | False)`. With `"auto"` (the default), frozen parameters are left out of the checkpoint when the model has any, and a signature of them is saved instead. Resuming with different frozen weights raises `CheckpointMismatchError` instead of continuing silently.
- Background saving: `Run(async_save=True)` (default) writes periodic checkpoints in a background thread after a fast copy to CPU memory. Budget, shutdown-signal and final saves wait for it and write synchronously. A failed background save raises at the next `run.step()`. `run.save(blocking=False)` starts a background save yourself.
- Checkpoint metadata (format version, Efficio version, mode, trainable-only flag, sizes), and report lines with the checkpoint mode, the last file size and the save time hidden by background saving.
- `benchmarks/checkpoint_sizes.py`: file size and save time for exact, light and trainable-only checkpoints, and the effect of a light resume.

### Changed

- Models with frozen parameters now get trainable-only checkpoints by default (`save_trainable_only="auto"`); pass `save_trainable_only=False` for the 0.1.x behaviour. Periodic saves now run in the background by default; pass `async_save=False` to write them synchronously as before.

### Compatibility

- Checkpoints written by Efficio 0.1.x load and resume unchanged. Exact-mode checkpoints of models without frozen parameters keep the 0.1.x layout (plus a metadata entry).

## [0.1.1] - 2026-10-27

Patch release, no new features.

### Changed

- The report no longer shows the setup-time tip or the "larger batch size" tip when training took less than 2 minutes. In a short run, setup is naturally a large share of the time and GPU utilization is measured over too little time, so these tips were misleading. Fixes the known issue listed for 0.1.0.

### Removed

- The `hub`, `light` and `monitor` extras. The features they were meant for (Hugging Face Hub storage, light checkpoints, monitoring) are not released yet. `pip install efficio[dev]` still works.

## [0.1.0] - 2026-10-27

First public release.

### Added

- `efficio.Run`: one object for a training run.
  - GPU-aware mixed precision (`run.autocast()`): fp16 with gradient scaling on a T4 and older GPUs, bf16 on Ampere and newer, fp32 on CPU.
  - Time budget (`Run(budget="2h")`): saves and stops before the time runs out; SIGTERM also triggers a save.
  - Checkpoints with model, optimizer, scheduler, gradient scaler, RNG states and the exact position in the epoch. Saves are atomic. Periodic and emergency saves.
  - Exact mid-epoch resume from the latest checkpoint (`run.prepare(...)`), including the same data order for shuffled DataLoaders (per-epoch seeded shuffle).
  - Gradient accumulation (`grad_accum_steps`), gradient clipping that works with fp16 and accumulation (`run.clip_grad_norm`), and a validation helper (`run.evaluate_mode()`).
  - Automatic DataLoader tuning (workers, pinned memory, persistent workers, prefetching), channels_last for convolutional models and `cudnn.benchmark` on the GPU.
  - GPU augmentation for any DataLoader through `run.loop(..., gpu_transform=...)` and `efficio.on_inputs`.
  - End-of-run report (`run.report()`): setup vs training time, data-wait, average GPU utilization, GPU minutes used, applied optimizations and tips.
- `efficio.GPULoader`: keeps a small dataset on the GPU and builds batches there (`from_torchvision`, `from_tensors`), with a seeded per-epoch order and instant resume skipping.
- `efficio.ImageCache` and `efficio.CachedImageDataset`: decode and resize a large image dataset once (parallel reads and decoding, resumable build) and train from a uint8 array.
- `efficio.gpu_augment`: batched GPU augmentation (random crop, resized crop, affine, horizontal flip, brightness and contrast, resize, normalize, to-float, repeat channels).
- Platform and GPU detection (`efficio.detect_platform`, `efficio.detect_hardware`) for Colab, Kaggle and local machines.

### Known issues

- The report's setup-time tip ("Setup took N% of this run...") also appears on very short runs, where setup is naturally a large share of a few seconds. It can be ignored there.

### Known limitations

- PyTorch only; single GPU; Python 3.10 or newer.
- On Windows and macOS, DataLoader workers are not added automatically.
- Planned features (light checkpoints, Hub and Drive storage, automatic batch size, pre-flight check, budget planner, auto-shutdown, multi-GPU, notifications, framework adapters) are not included; their modules raise `NotImplementedError`.

[0.1.1]: https://github.com/HuNTer8272/Efficio/releases/tag/v0.1.1
[0.1.0]: https://github.com/HuNTer8272/Efficio/releases/tag/v0.1.0
