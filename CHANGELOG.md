# Changelog

All notable changes to Efficio are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

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

[0.1.0]: https://github.com/HuNTer8272/Efficio/releases/tag/v0.1.0
