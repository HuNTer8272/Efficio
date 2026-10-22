# Progress

## 2026-10-17

**What changed**
- Phase 1 was verified on Colab (1x Tesla T4, fp16) and Kaggle (2x Tesla T4, fp16). We checked platform and GPU detection, fp16 mixed precision, the budget stop and exact mid-epoch resume. Both platforms ended at the identical final step count (31,400) after resuming.
- Fixed the GPU resume bug. Checkpoints are now loaded with `map_location="cpu"`, and RNG state is kept on CPU.
- Rewrote README.md for v1. It now says local GPUs gain speed and crash safety, not extra quota, and it lists each feature as Available or Planned.
- Added CLAUDE.md with the project context and the rules for working in this repo.
- Removed the duplicate `efficio/benchmarks/` folder. The real one is `benchmarks/` at the repo root.
- Added the ResNet-18 / CIFAR-10 benchmark: `benchmarks/common.py`, `resnet_cifar_baseline.py`, `resnet_cifar_efficio.py`, `compare.py` and `README.md`.
- Added `data/` to `.gitignore`.

**Files touched**
README.md, CLAUDE.md, PROGRESS.md, .gitignore, benchmarks/*, efficio/benchmarks/ (deleted)

**How it was tested**
- Phase 1 was tested on Colab and Kaggle, as described above.
- The benchmark scripts were only syntax-checked (`python -m py_compile`). We also confirmed that `resnet_cifar_baseline.py` and `common.py` do not import efficio.

**Still needs a GPU test**
- The ResNet benchmark has NOT been run on a GPU yet. It needs to run on Colab and Kaggle, with the result JSON files committed.

**Open questions**
- On Kaggle, Efficio still trains on only one of the two T4 GPUs. Multi-GPU is planned for Phase 4.

## 2026-10-17 (Phase 2 step 1, branch feat/profiler-dataloading)

**What changed**
- Profiler (`efficio/report/profiler.py`, no torch import). It tracks setup time (from Run creation to the first trained batch), training time, and data-wait time per step (wall-clock time spent waiting for the next batch, measured in `Run.loop`). It also runs a GPU utilization sampler: nvidia-smi on GPU 0 every 1 s, only on CUDA, and it never raises.
- DataLoader tuning (`efficio/speed/dataloader.py`). `optimize_loader` rebuilds the loader with more workers (CPU cores, capped at 8), pin_memory on CUDA, persistent workers and prefetch_factor 4. It keeps the dataset, batch_size, sampler, collate_fn, drop_last, worker_init_fn and generator. Custom batch_sampler loaders, IterableDatasets and non-DataLoaders are left unchanged. `Run(optimize_data=True)` applies it by default.
- Automatic speedups. cudnn.benchmark is turned on with CUDA. channels_last (`efficio/speed/memory_format.py`) is applied to conv models on the GPU, and 4-D float inputs are converted too. It is set with `Run(channels_last="auto"|True|False)`.
- The report now adds: setup vs training time, data-wait %, average GPU utilization, GPU session minutes, the optimizations applied, and plain-language tips.
- Benchmark: `resnet_cifar_efficio.py` has a new `--no-optimize-data` flag, and those runs are saved as `efficio-no-data-opt`. `compare.py` prints a speedup for every Efficio variant. The baseline script was not changed.
- Fixed `detect_platform()`, which crashed with ModuleNotFoundError when the `google` package wasn't installed. Because of that bug, `efficio.Run()` failed on a clean local machine.
- README: added four rows marked "Available, needs GPU verification".

**Files touched**
efficio/run.py, efficio/report/profiler.py, efficio/speed/dataloader.py (new), efficio/speed/memory_format.py (new), efficio/env/detect.py, tests/test_speed_and_report.py (new), benchmarks/resnet_cifar_efficio.py, benchmarks/compare.py, benchmarks/README.md, README.md, PROGRESS.md

**How it was tested**
- On CPU (Windows, torch 2.14.1+cpu), pytest gives 15 passed. The tests cover:
  - optimize_loader keeps dataset length, batch count, batch_size, drop_last and shuffle behavior, and returns custom batch_sampler, IterableDataset and non-DataLoader loaders unchanged
  - channels_last is skipped on CPU and for models without Conv2d
  - profiler percentages and tips, using fake timings
  - mid-epoch resume through the optimized loader with 2 workers gives the same step count and the same weights as an uninterrupted run (shuffle=False)
- Ran `examples/quickstart.py` on CPU to check the new report lines.

**Still needs a GPU test**
- On Colab and Kaggle, check:
  - the dataloader log line and the worker counts
  - that pin_memory and channels_last are applied
  - that nvidia-smi sampling gives sensible utilization numbers
  - the report and tips
  - that resume still works with persistent workers
- Run the ResNet benchmark three ways (baseline, efficio, `--no-optimize-data`) and commit the result JSON files.

**Open questions**
- Existing issue, also on main: with `shuffle=True`, mid-epoch resume is not data-exact. The checkpoint stores the RNG state from mid-epoch, so on resume the epoch is reshuffled with a different permutation before batches are skipped. Some samples in that epoch are trained twice and others are skipped. The step count still matches, and that was all the Phase 1 test checked. A fix would need the epoch-start RNG state saved in the checkpoint (checkpoint code, Person A). Until then, the README's "Exact mid-epoch resume" claim only holds for unshuffled data.
- With persistent workers, the shuffle order in epochs after the first differs from the user's original loader. Each epoch is still a full random permutation, but it is not the same one the original loader would have produced.
- On Windows/macOS ("spawn" start method), workers are not added automatically. Extra workers need a picklable dataset and an `if __name__ == "__main__":` guard, so the user has to opt in by setting num_workers. Colab and Kaggle (Linux, "fork") get extra workers automatically.

## 2026-10-21 (Phase 2 step 2, branch feat/profiler-dataloading)

**Step 1: the error from the last Colab run**
No traceback was available. `pytest -q` and both quickstart commands (`--budget 20s`, then resume) pass locally on Windows (CPU) and on Linux/WSL (CPU, "fork" workers like Colab/Kaggle), including a third run after the training had already finished. Problems found by reviewing the branch, all fixed:
- `test_run_skips_channels_last_on_cpu` fails on every GPU machine. It used the real hardware, so on Colab, Run picked CUDA and turned channels_last on. This is the most likely cause of the Colab error. Run-based tests now force CPU hardware with a `cpu_only` fixture, so they check the same thing everywhere.
- Calling `run.loop()` a second time with the same Run (re-running a notebook cell, or a second training phase) trained one batch and then stopped. The old "finished" stop reason was never cleared. It is now reset when a new loop starts.
- The SIGTERM handler was installed in `Run.__init__` and never removed. In a notebook kernel, a later SIGTERM only set a flag, so the kernel did not shut down. The handler is now installed only while `run.loop()` runs, and the previous one is restored afterwards.
- The GPU utilization sampler: it now stops trying if nvidia-smi is missing (it used to retry every second). `stop()` waits at most 1 s, so a slow nvidia-smi can't hold up the end of training. It starts at the first trained batch, after DataLoader workers have been forked (forking with a running thread can deadlock), so setup and resume skipping are no longer part of the utilization average.
- `run.optimizations` no longer repeats entries when `loop()` is called more than once.
- Checked and found fine: report percentages with a very short run or a run resumed at the very end (no division by zero, now covered by a test), and resume with persistent workers (tested on Linux).

**Kaggle hang / slow CIFAR-10 download**
- `benchmarks/common.py`: `prepare_cifar10` looks under `/kaggle/input` (up to 4 levels deep) for a `cifar-10-batches-py` folder or `cifar-10-python.tar.gz` and copies or extracts it into the data dir, so torchvision skips the download. It prints the source used: existing copy, copied folder, extracted archive, or download. Only the data location changes, so it is fair to both scripts.
- `--data-dir` (default `./data`) added to both benchmark scripts. That is the only change to the baseline script.

**GPU data pipeline (GPULoader)**
- `efficio/speed/gpu_data.py`: `GPULoader.from_tensors` / `from_torchvision` keeps the dataset on the GPU (images as uint8, converted to float per batch), with a per-epoch permutation from `seed + epoch` (`set_epoch`), `set_start_batch(n)` for instant resume skipping, `__len__`, `batch_size`, and a memory check (refuses datasets above 30% of free GPU memory). It works on CPU too. `from_torchvision` raises a clear error without `.data`/`.targets`, and warns when the dataset has its own transform (which GPULoader ignores).
- `efficio/speed/gpu_augment.py`: batched `random_crop` (zero padding, like torchvision RandomCrop), `random_horizontal_flip`, `normalize`, `Compose`, plus small `RandomCrop` / `RandomHorizontalFlip` / `Normalize` classes. Randomness is per sample and comes from the global torch RNG, which checkpoints save.
- `Run.loop`: calls `set_epoch` and `set_start_batch` when the loader has them, skips `optimize_loader` for a GPULoader, and still applies channels_last to its batches. `efficio.GPULoader` is exported.
- Report: when data-wait is above 10% and the training set is a torchvision dataset that fits on the GPU, the tip suggests GPULoader with a 3-line example.
- Benchmark: `resnet_cifar_efficio.py --gpu-data` builds a GPULoader from the same train set, with the same augmentation, batch size 128, seed 0 and drop_last=True. It is saved as `efficio-gpu-data` (or `efficio-gpu-data-no-data-opt` together with `--no-optimize-data`). The test set keeps the normal loader.

**Files touched**
efficio/speed/gpu_data.py (new), efficio/speed/gpu_augment.py (new), efficio/run.py, efficio/report/profiler.py, efficio/__init__.py, tests/test_gpu_data.py (new), tests/test_speed_and_report.py, benchmarks/common.py, benchmarks/resnet_cifar_baseline.py (`--data-dir` only), benchmarks/resnet_cifar_efficio.py, benchmarks/README.md, README.md, PROGRESS.md

**How it was tested**
- pytest: 37 passed on Windows (torch 2.14.1+cpu) and on Linux/WSL (torch 2.14.1+cpu). New tests cover: GPULoader batch count and shapes, every sample exactly once per epoch, drop_last, same permutation for the same seed and epoch and a different one for other epochs, set_start_batch, input layouts, the memory check, from_torchvision with a fake dataset, the augmentation functions, and resume through `Run.loop` with a shuffled, augmented GPULoader. That resume test ends at the same step count with identical weights. Also covered: the SIGTERM handler restore, the report after resuming a finished run, a second `loop()` call, and the sampler without nvidia-smi.
- On Linux/WSL with a fake CIFAR-10 in the real file format (the real download ran at about 40 kB/s): `prepare_cifar10` picks the attached archive, the attached folder, an existing copy, or falls back to download. Without augmentation, GPULoader batches are identical to torchvision's ToTensor+Normalize (max difference 0.0). With augmentation, the per-channel mean and std match torchvision's pipeline to within 0.001.
- CPU smoke run on Linux/WSL of the baseline and of `resnet_cifar_efficio.py` with no flag, `--gpu-data`, and `--gpu-data --no-optimize-data` on the fake data: all ran end to end. Each saved its JSON under the right run name, and `compare.py` read them. The timings are CPU-only on fake data and are not benchmark results, so they were not committed.

**Still needs a GPU test**
- Colab and Kaggle: `pytest -q`, both quickstart commands.
- `resnet_cifar_efficio.py --gpu-data` on Colab and Kaggle: epoch time, GPU utilization, data-wait in the report, test accuracy close to the DataLoader run, and channels_last applied to GPULoader batches. Commit the result JSON files.
- Kaggle: attach a CIFAR-10 dataset and check that the "CIFAR-10 source" line reports the copied or extracted copy.
- The GPULoader tip in the report on a run with high data-wait.
- Resume with GPULoader on a GPU (stop with a budget, run again).
- SIGTERM during training on Kaggle still saves at the next step.

**Open questions**
- GPULoader shuffles with its own seeded generator, so the sample order differs from the baseline DataLoader's order. Accuracy should be statistically the same, but the runs are not sample-for-sample identical.
- With GPULoader, mid-epoch resume is data-exact (same order and same augmentation, checked by the test). With a shuffled DataLoader it still is not (see the 2026-10-17 entry).
- Only `.data`/`.targets` datasets are supported by `from_torchvision`. SVHN (`.labels`) and ImageFolder need `from_tensors`.

## 2026-10-21 (Colab GPU verification of Phase 2, branch feat/profiler-dataloading)

**What changed**
- README: DataLoader tuning, channels_last, GPULoader, profiler and the improved report are now marked Available (verified on a Colab T4). Added the Colab ResNet-18 results under "Measuring results".
- Committed the three Colab result files in `benchmarks/results/`. The Colab runtime was reset before the files could be downloaded, so they were rebuilt from the JSON the scripts printed during the run. `save_results` prints exactly what it writes, so the values are the measured ones and `compare.py` reproduces the printed table.

**Files touched**
README.md, PROGRESS.md, benchmarks/results/baseline_colab_20261021_084307.json, benchmarks/results/efficio_colab_20261021_084501.json, benchmarks/results/efficio-gpu-data_colab_20261021_084617.json

**How it was tested (Colab, 1x Tesla T4, torch 2.11.0+cu130, commit 43f96ed)**
- `pytest -q`: 37 passed. The test that used to fail on GPUs now passes. 2 warnings come from tests that create 3-4 workers on the 2-CPU machine.
- Quickstart: fp16 detected, budget stop at step 2490, resumed from (15, 135), finished at epoch 200, step 31400 (the same step count as on CPU). No error after the report, so the unexplained error from the last Colab run did not come back.
- SIGTERM sent to quickstart: "stopping: shutdown signal", checkpoint saved, report says "ended because: shutdown signal".
- ResNet-18 / CIFAR-10, 5 epochs:
  - baseline: 47.6 s per epoch, 75.6 epochs/hour, 98% GPU utilization, 85.37% accuracy
  - efficio: 20.7 s, 173.6 epochs/hour, 54% utilization, 22% data-wait, 85.57%, 2.30x
  - efficio-gpu-data: 13.3 s, 270.4 epochs/hour, 96% utilization, 5% data-wait, 85.59%, 3.58x
  - The efficio run's report suggested GPULoader with the 3-line example, as intended. In the gpu-data run, channels_last was applied to GPULoader batches.
- GPULoader resume on the GPU (CIFAR-shaped random data, 60 s budget): the first run stopped at step 836 (epoch 2, batch 56). The second run resumed from (2, 56) and finished at epoch 3, step 1170 (3 x 390).

**Still needs a GPU test**
- Kaggle (2x T4): pytest, quickstart, the benchmark with all three runs, and the attached CIFAR-10 copy (the "CIFAR-10 source" line should report a copied folder or extracted archive).

**Open questions**
- The CIFAR-10 download also took 42 min 43 s on Colab (66.5 kB/s), using GPU quota before training started. Options: download once on a CPU runtime into Google Drive and pass `--data-dir`, or have `prepare_cifar10` also look in a mounted Drive folder.
- On short runs (about 30 s), GPU utilization reads lower (82-89%) because cuDNN warm-up is included, so the "larger batch size" tip can appear. Consider skipping that tip for very short runs.

## 2026-10-22 (Kaggle GPU verification, branch feat/profiler-dataloading)

**What changed**
- Committed the three Kaggle result files in `benchmarks/results/`. As with Colab, they were rebuilt from the JSON the scripts printed, and `compare.py` reproduces the printed table.
- README: added the Kaggle results table and explained why GPULoader gives no extra speed on Kaggle. "Tested on" now covers both platforms.

**Files touched**
README.md, PROGRESS.md, benchmarks/results/baseline_kaggle_20261021_184650.json, benchmarks/results/efficio_kaggle_20261021_184817.json, benchmarks/results/efficio-gpu-data_kaggle_20261021_184943.json

**How it was tested (Kaggle, 2x Tesla T4, torch 2.11.0+cu128, commit 04b8ace)**
- `pytest -q`: 37 passed. 20 DeprecationWarnings about fork() in a multi-threaded process come from torch's own threads during the DataLoader tests; the same warnings appear on Linux CPU.
- Quickstart: "kaggle: 2x Tesla T4", fp16, 4 DataLoader workers. Budget stop at step 2516, resumed from (16, 4), finished at epoch 200, step 31400 (the same as on Colab and CPU).
- ResNet-18 / CIFAR-10, 5 epochs:
  - baseline: 44.8 s per epoch, 80.3 epochs/hour, 98% utilization, 85.98% accuracy
  - efficio: 14.7 s, 244.3 epochs/hour, 89% utilization, 2% data-wait, 85.55%, 3.04x
  - efficio-gpu-data: 14.7 s, 244.3 epochs/hour, 91% utilization, 5% data-wait, 85.52%, 3.04x
- With 4 CPU cores, the DataLoader is not the bottleneck on Kaggle, so GPULoader gives the same speed. The report correctly did not suggest GPULoader (data-wait was 2%).

**Still needs a GPU test**
- The attached-dataset path in `prepare_cifar10` is still unverified on Kaggle. No CIFAR-10 dataset was attached (`find /kaggle/input` found nothing), so the benchmark downloaded it again (48 min 29 s at 58.6 kB/s). Next Kaggle session: attach a dataset with `cifar-10-python.tar.gz` or `cifar-10-batches-py` and check the "CIFAR-10 source" line.
- GPULoader resume and the SIGTERM test were only run on Colab.

**Open questions**
- Only one of Kaggle's two T4s is used. Using both (multi-GPU, Phase 4) is the next big speedup on Kaggle.
- The CIFAR-10 download cost 43-48 min of GPU session on both platforms. Downloading once into Drive (Colab) or a private Kaggle dataset would avoid this.

## 2026-10-22 (Phase 2 wrap-up, branch feat/profiler-dataloading)

**What changed**
- Data-exact resume with a shuffled DataLoader. `seeded_shuffle` (`efficio/speed/dataloader.py`) rebuilds a loader that uses a plain RandomSampler so the order comes from its own generator, seeded with base seed + epoch (as in GPULoader). A second generator (base seed + epoch + offset) supplies the DataLoader worker seed, so neither draws from the global RNG. `Run.loop` reseeds both before each epoch.
  - The base seed is the user's DataLoader generator seed if they gave one, otherwise `torch.initial_seed()`, so nothing is drawn from the RNG.
  - The base seed is stored in the checkpoint under a new optional `extra` dict (`CheckpointManager.save(..., extra=...)`; `CheckpointManager.extra` after `load`). Older checkpoints without it still load. interfaces.py was not changed.
- When resuming mid-epoch through a DataLoader, `Run.loop` restores the RNG state after skipping the already-trained batches, so random CPU augmentation in the main process (num_workers=0) also continues exactly.
- Shared `_rebuild` helper in `speed/dataloader.py`, used by `optimize_loader` and `seeded_shuffle`.
- README:
  - The resume row now says exactly when the data order is reproduced.
  - New row for shuffled-DataLoader resume (needs GPU verification).
  - Roadmap: Phase 2 done so far / remaining. The data cache moved from Phase 3 to Phase 2.
  - Note that the headline numbers are single 5-epoch runs, to be averaged over 3 runs before release.
- Benchmark comment corrected: the Efficio run still starts from the same model as the baseline, but its shuffle order now differs.
- CLAUDE.md: permanent rule against AI attribution in commits and PRs.
- Checked the two Kaggle result files: their averages match only by coincidence. The epoch time lists differ: [20.414, 15.197, 15.178, 14.358, 14.205] vs [19.102, 14.772, 15.596, 14.59, 13.976]; the sums of epochs 2-5 are 58.938 s vs 58.934 s. Utilization, accuracy and memory differ too. No numbers were changed.

**Files touched**
efficio/speed/dataloader.py, efficio/checkpoint/manager.py, efficio/run.py, tests/test_speed_and_report.py, benchmarks/resnet_cifar_efficio.py (comment only), README.md, CLAUDE.md, PROGRESS.md

**How it was tested**
- pytest, CPU, Windows (torch 2.5.1+cpu): 41 passed. New tests:
  - shuffled DataLoader, stop mid-epoch, resume: the same batches in the same order and identical weights to an uninterrupted run, both with num_workers=0 plus a random CPU augmentation and with the optimized 2-worker persistent loader
  - seeded_shuffle order per seed and epoch
  - an old checkpoint without `extra` still loads
- With the fix switched off (seeded_shuffle and the RNG restore disabled), both resume cases fail, so the tests detect the old behavior.

**Still needs a GPU test**
- Shuffled-DataLoader resume on Colab/Kaggle (stop with a budget mid-epoch, resume, compare with an uninterrupted run).

**Open questions**
- Random augmentation inside DataLoader workers (num_workers > 0) is still not reproduced exactly after a resume: persistent workers keep their own RNG across epochs. The order is exact; only worker-side augmentation differs. GPU augmentation (GPULoader, or gpu_transform on the next branch) uses the global RNG and is exact.
- CLAUDE.md's roadmap still lists the data cache under Phase 3; the README now lists it under Phase 2.

## 2026-10-22 (large image datasets, branch feat/large-image-support)

**What changed**
- `efficio/data/image_cache.py` (new; no torch import at module level, so TensorFlow/Keras support can reuse it):
  - `ImageCache.build(paths, labels, out_dir, size, grayscale, workers)` decodes and resizes every image once with a process pool into one uint8 array file (`images.npy`, N x H x W x C), plus `labels.npy` (int class ids, or a float multi-hot array) and `meta.json` (size, count, channels, label shape, SHA-1 of the path list).
  - The build is resumable: each chunk is flushed to disk before it is recorded as done in `progress.json` (atomic write), so a rerun continues with the missing chunks. A finished cache with matching metadata returns immediately; changed settings or paths trigger a rebuild.
  - The progress line shows images/s and the time left. The cache size is printed when finished, with a storage tip per platform (Drive on Colab, output or private dataset on Kaggle). Nothing is uploaded.
  - `CachedImageDataset` returns (uint8 C x H x W tensor, label) from the memory-mapped file. The arrays are opened lazily in each process, so DataLoader workers don't copy them, and each sample is copied out of the read-only map.
- `Run.loop(loader, epochs, gpu_transform=None)`: the transform runs on each batch after it is on the device (no gradients). channels_last is applied after the transform. `efficio.on_inputs(t)` applies `t` to the inputs of a tuple/list batch or to `pixel_values` / `image` of a dict batch.
- `speed/gpu_augment.py`: ToFloat, RepeatChannels, Resize, RandomResizedCrop, RandomAffine (rotation, translation and scale through affine_grid, correct for non-square images), RandomBrightnessContrast. They are batched, random per sample, and use the global torch RNG.
- `run.clip_grad_norm(max_norm)`: unscales through the GradScaler before clipping, and only acts on the micro-batch where the optimizer steps (gradient accumulation). It returns the norm, or None while gradients are still accumulating.
- `run.evaluate_mode()`: eval mode, no_grad and the run's autocast; restores the previous train/eval mode.
- Report: when data-wait is above 10% and the dataset decodes image files every epoch (ImageFolder-like `samples`/`imgs`/`image_paths`, or "image" in the class name; not CachedImageDataset), the tip suggests ImageCache with an example. The GPULoader tip still takes priority when the data fits on the GPU. The setup tip now points to ImageCache instead of "a data cache is planned".
- `examples/large_image_dataset.py`: image folder + CSV, cache build, CachedImageDataset + DataLoader, GPU augmentation, a pretrained torchvision ResNet/DenseNet, BCEWithLogitsLoss (multi-label) or cross-entropy, clip_grad_norm, evaluate_mode with mean AUROC, budget and resume. `--fake` generates 300 random grayscale PNGs.
- `efficio.ImageCache`, `efficio.CachedImageDataset` and `efficio.on_inputs` are exported. Added a `pillow` optional extra. `image_cache/` and `fake_images/` are git-ignored.
- README: new rows (needs GPU verification), a "Large image datasets" section, clip_grad_norm and evaluate_mode in the Quick start, Phase 2 progress.

**Files touched**
efficio/data/__init__.py (new), efficio/data/image_cache.py (new), efficio/speed/gpu_augment.py, efficio/run.py, efficio/report/profiler.py, efficio/__init__.py, examples/large_image_dataset.py (new), tests/test_image_cache.py (new), tests/test_large_image_features.py (new), tests/test_speed_and_report.py, README.md, PROGRESS.md, pyproject.toml, .gitignore

**How it was tested**
- pytest: 63 passed on Windows (torch 2.5.1+cpu, spawn workers) and on Linux/WSL (torch 2.14.1+cpu, fork workers). New tests cover:
  - the cache build on generated PNGs (single-label RGB and multi-label grayscale, pixel-exact against PIL)
  - an interrupted build that resumes and decodes only the missing chunks, with a result identical to a clean build
  - a finished cache that returns immediately, and a rebuild when the settings change
  - the process-pool build, and CachedImageDataset in a DataLoader with 2 workers and through optimize_loader
  - each new augment op: shape, dtype, reproducible with the same seed, random per sample, exact geometry for 90 deg / 180 deg / 1-pixel shift
  - on_inputs
  - gpu_transform in Run.loop with a shuffled DataLoader: resume gives identical weights
  - clip_grad_norm with a disabled scaler, with gradient accumulation, and with an enabled GradScaler (same result as unscaled fp32 clipping)
  - evaluate_mode, and the image-file detection behind the tip
- `examples/large_image_dataset.py --fake` on Linux/WSL CPU: first run built both caches and stopped on a 50 s budget at step 40; the second run reused the caches, resumed from (9, 4) and finished at step 120.

**Still needs a GPU test**
- ImageCache build speed (images/s) on Colab (2 cores) and Kaggle (4 cores) with real PNG/JPEG files; training speed and GPU utilization with CachedImageDataset + gpu_transform vs a normal ImageFolder DataLoader with CPU transforms.
- gpu_transform with fp16 autocast and channels_last on a T4; clip_grad_norm with the real fp16 GradScaler; evaluate_mode speed.
- The ImageCache tip on a real image-folder run with high data-wait.
- Reading a cache from Google Drive (much slower than local disk) vs copying it to /content first.

**Open questions**
- The images are resized to the target size without keeping the aspect ratio. Fine for most X-ray/medical datasets (often square), but an aspect-preserving resize + center crop option may be useful.
- RandomResizedCrop has no antialiasing when it shrinks a large crop (grid_sample). Cache at the training size so crops are only slightly scaled.
- A 100k x 224 x 224 grayscale cache is about 5 GB (RGB about 15 GB). On Colab, building into /content and then copying to Drive is much faster than writing to Drive directly.

## 2026-10-22 (Kaggle GPU run of feat/large-image-support)

**What was run**
Kaggle notebook, 2x Tesla T4, commit ced9d17: `pytest -q`, then `examples/large_image_dataset.py --help` and `--fake`.

**Results**
- pytest: 63 passed (30 DeprecationWarnings about fork() in a multi-threaded process from torch's own threads in the DataLoader/process-pool tests; harmless).
- The example ran end to end on the GPU with no errors:
  - ImageCache built the train (270 images, 1.1 MB) and val (30 images, 0.1 MB) caches and printed the Kaggle storage tip.
  - Training ran with fp16 + GradScaler, channels_last, cudnn.benchmark, `gpu_transform` (ToFloat, RepeatChannels, RandomResizedCrop, RandomAffine, flip, brightness/contrast, Normalize), `clip_grad_norm` with the real fp16 GradScaler on CUDA, and `evaluate_mode`. It finished at epoch 10, step 40.
  - Validation mean AUROC 0.441 (random labels, so about chance, as expected).
- What this does not show: speed (tiny 64x64 random images; 7% GPU utilization because the GPU had almost no work, so the "larger batch size" tip is expected), resume with gpu_transform on the GPU, the clipping values in real units on CUDA, the ImageCache tip, and reading a cache from Drive. The new features stay "Available, needs GPU verification".

**Problem found**
- The notebook cloned with `https://{token}@github.com/...`, which leaves the token in `.git/config` under /kaggle/working, and Kaggle publishes that folder on Save Version. Added a rule to CLAUDE.md: notebook clone instructions must reset the remote URL without the token right after cloning, and token checks look for both `github_pat_` and `ghp_`.

## 2026-10-22 (image-folder benchmark, branch feat/large-image-support)

**What changed**
- `benchmarks/image_common.py` (no efficio import): fake data (default 5,000 grayscale 512x512 PNGs, smooth random content with fine noise, written once with a process pool to ./data/image_folder) or a real dataset via --images/--labels (same CSV format as the example), a fixed 90/10 split, ResNet-18 with a multi-label head, AdamW, the shared augmentation settings, mean AUROC and per-label accuracy.
- `benchmarks/image_folder_baseline.py`: plain PyTorch fp32. A Dataset decodes every PNG and runs the torchvision transforms on the CPU each epoch (2 workers). It measures epoch time, data-wait (time in next()), GPU utilization, validation accuracy and AUROC.
- `benchmarks/image_folder_efficio.py`:
  - builds an ImageCache once and records `cache_build_s` (0 when reused)
  - trains through Efficio with the same augmentation on the GPU (`gpu_transform`)
  - budget stop and resume: after 2 full epochs the budget is set to run out halfway through epoch 3, the run saves and stops, and a fresh Run with new model/optimizer objects resumes and finishes
  - records `resume` (where it stopped and resumed, final vs expected step), `epochs_all_s` (with the split epoch marked incomplete), data-wait over both phases and the optimizations
- `benchmarks/compare.py`: a benchmark column (older CIFAR files default to `resnet18-cifar10`), data-wait and cache-build columns, and speedups per benchmark and platform against the run whose name ends in "baseline". The CIFAR speedups are unchanged (2.30x / 3.58x Colab, 3.04x Kaggle).
- benchmarks/README.md: an image-folder section. CLAUDE.md: the token rule above.

**Files touched**
benchmarks/image_common.py (new), benchmarks/image_folder_baseline.py (new), benchmarks/image_folder_efficio.py (new), benchmarks/compare.py, benchmarks/README.md, CLAUDE.md, PROGRESS.md

**How it was tested**
- Linux/WSL CPU smoke test with small settings (300 fake PNGs at 128x128, size 64, batch 16, 4 epochs): both scripts ran and saved results.
  - The Efficio run stopped on the budget at epoch 3, batch 10 (step 42); a fresh Run resumed from (2, 10) and finished at step 64 = expected 64 (resume.ok true). The split epoch was excluded from the average.
  - A second Efficio run reused the cache (cache_build_s 0.0).
  - compare.py grouped the results per benchmark.
- The CPU timings are not meaningful (no GPU, tiny images) and were not committed.

**Still needs a GPU test**
- Run both scripts with --fake (5,000 x 512x512) on Colab and on Kaggle and commit the result files: epoch time, GPU utilization, data-wait, cache build time, and the resume check on the GPU.
- Optionally a real attached dataset (--images/--labels, with --limit to keep the baseline short).

**Open questions**
- On fake data the PNGs were just written, so the OS file cache may hold them, which makes the baseline's file reads faster than with a real dataset in /kaggle/input or on Drive. The decode/resize/augment cost on the CPU remains, and that is what is being compared.
- The baseline uses the same 2 DataLoader workers as the CIFAR baseline. On Kaggle (4 cores) a user might set 4; the Efficio run tunes this itself.

## 2026-10-22 (parallel reads in ImageCache.build, branch feat/large-image-support)

**Kaggle finding (real data: NIH Chest X-rays, 1024x1024 PNGs from /kaggle/input)**
- `ImageCache.build` ran at only **2 images/s**.
- Diagnosis: sequential file reads give 2.5 files/s (high per-file latency on Kaggle's dataset storage); decode + resize gives 55 images/s per core, about 200/s on 4 cores. A 32-thread parallel read test gave **71.5 files/s (28.6 MB/s)**, 28x faster. The build is limited by I/O latency, not by the CPU.

**What changed**
- `ImageCache.build(..., io_threads=32)`: a thread pool reads file bytes ahead of decoding, and worker processes decode groups of 16 images from those bytes (PIL via io.BytesIO), so reading and decoding overlap.
  - At most 4 x io_threads files are held between reading and writing.
  - Chunks are still recorded as done only after they are flushed, so the build stays resumable.
  - Without a process pool (workers 0/1), each group is written right after it is decoded. Otherwise an interrupt could lose decoded but unwritten work, which the interrupt test caught on Linux.
- Progress line: overall images/s, read files/s and decode capacity (images per second of decode work x workers). A one-time tip when reads stay below half the decode capacity after 10 s: storage is the bottleneck, try a higher io_threads (2x the current value).
- One-time note when a cache is built on Colab/Kaggle with a CUDA GPU: building needs no GPU, so build it in a CPU-only session, save it, then train in a GPU session. Information only; nothing else changes. CUDA is checked with a lazy torch import, so the module stays torch-free at import.
- `cache.build_stats` (seconds, images/s, read files/s, decode capacity, io_threads, workers, max in flight) after a build; None when a finished cache was reused.
- `--io-threads` in `benchmarks/image_folder_efficio.py` (whose result file now records `cache_build_stats`) and in `examples/large_image_dataset.py`.
- README: io_threads and the GPU-quota note in "Large image datasets".

**Files touched**
efficio/data/image_cache.py, tests/test_image_cache.py, benchmarks/image_folder_efficio.py, examples/large_image_dataset.py, README.md, PROGRESS.md

**How it was tested**
- pytest: 70 passed on Windows (torch 2.5.1+cpu) and 3 times in a row on Linux/WSL (torch 2.14.1+cpu). New or updated tests:
  - io_threads=1, io_threads=8 and io_threads=8 with 2 decode processes give identical caches, equal to decoding from the file paths
  - interrupt and resume (decodes only the missing chunks, same result as a clean build)
  - the in-flight limit (never more than 4 x io_threads files held, and it does read ahead up to the limit)
  - the storage-bottleneck tip
  - the GPU note for kaggle/colab/local and with/without CUDA, shown at most once
- Byte-identical output: the previous implementation (git HEAD before this change) and the new one built caches from 40 PNGs (RGB, grayscale, RGBA, palette, 16-bit) in RGB and grayscale mode, and the images.npy files were byte-identical.
- Simulated slow storage on Linux (each read delayed by 50 ms, 300 images, 2 decode processes): io_threads=1 gave 18-19 images/s, io_threads=32 gave about 570 images/s. This is only a simulation of the latency, not a Kaggle measurement.
- Smoke runs of the image-folder benchmark (resume check OK, build stats saved) and of the example with --io-threads.

**Still needs a GPU/Kaggle test**
- Rerun the build on NIH Chest X-rays from /kaggle/input and record the images/s after the fix (to be filled in here when measured), then the image-folder training runs.

**Open questions**
- If 32 threads are still latency-bound on Kaggle, io_threads=64 may help; the progress tip will say so. Very high values mostly add memory (4 x io_threads files in flight, about 1 MB each for 1024x1024 PNGs).

## 2026-10-22 (NIH Chest X-ray results on Kaggle, out-of-order cache writes, branch feat/large-image-support)

**Kaggle run (commit d784423, Tesla T4, NIH Chest X-rays from /kaggle/input)**
5,000 randomly sampled 1024x1024 PNGs (seed 0; 4,500 train / 500 val), 14 findings multi-hot, ResNet-18 at 224, batch 64, 4 epochs.
- **Cache build: 2 -> 48 images/s.** Before the parallel reads, the build ran at 2 images/s. With io_threads=32 the train cache (4,500 images) took 93.8 s at 48 images/s; both caches took 105.1 s (215.6 + 24.0 MB). For comparison, a 32-thread raw read test earlier gave 71.5 files/s, and sequential reads 2.5 files/s.
  - Decode capacity was 232 images/s, so storage was still the limit, and the progress tip correctly suggested io_threads=64.
  - `max_in_flight` reached the 128 limit. Groups were processed strictly in order, so a slow file held back the files behind it; this is fixed below.
- The one-time GPU-quota note printed (GPU session), as intended.
- **Resume on real data, on the GPU:** the budget stopped the run at epoch 3, batch 36 (step 176 = 2 x 70 + 36). A fresh Run resumed from (2, 36) and finished at step 280 = expected 280 (resume.ok true).
- Efficio training: 5.09 s per epoch (resume run) and 5.13 s (rerun with the cache reused in 0 s), GPU utilization 76-77%, data-wait 0.3%.
- Baseline (fp32, CPU decoding, 2 workers): 54.65 s per epoch, GPU utilization 23.8%, data-wait 87.9%.
- Validation AUROC: baseline 0.670, Efficio 0.645 / 0.645. Per-label accuracy is about 95% for all runs because most labels are 0, so it doesn't say much.
- compare.py: 701.6 vs 65.9 epochs/hour (10.65x) in this setup.
  - Caveats: the baseline used 2 workers on 4 cores; it read warm files (after the cache build); single 4-epoch runs; AUROC differences at this scale are noise.
  - The README shows the table with these caveats and does not quote the ratio as a general claim.
- Saved as benchmarks/results/imagefolder-efficio_kaggle_20261022_151028.json, imagefolder-efficio_kaggle_20261022_151159.json and imagefolder-baseline_kaggle_20261022_151551.json, rebuilt from the printed JSON. The 151159 file predates the --no-resume-test fix below and still has stopped_at [4, 0, 280] and ok true; those fields carry no resume information there.

**What changed**
- ImageCache.build writes groups as they finish instead of strictly in order:
  - a group goes to decoding as soon as its own files are read, and decoded groups are written in completion order
  - each group writes only its own rows, and a chunk is recorded as done only when all its images are written and flushed, so resume stays correct and the output byte-identical
  - reads are still submitted in file order, within the same 4 x io_threads in-flight limit
- `image_folder_efficio.py --no-resume-test`: the resume fields stay empty (stopped_at/resumed_from [], final/expected step and ok None), and no "resume check" line is printed.
- `image_folder_baseline.py --workers N` (default 2, so earlier results stay comparable). Runs with another worker count are saved as `imagefolder-baseline-<N>workers`. Both scripts record `num_workers`.
- compare.py: every run whose name contains "baseline" is a baseline; Efficio runs are compared against each baseline.
- README:
  - ImageCache, GPU augmentation for any DataLoader and evaluate_mode marked Available (verified on the Kaggle T4 with real data; Colab not yet run)
  - clip_grad_norm, shuffled-resume order, and the ImageCache report tip stay "needs GPU verification"
  - NIH results table with the caveats

**Files touched**
efficio/data/image_cache.py, tests/test_image_cache.py, benchmarks/image_folder_efficio.py, benchmarks/image_folder_baseline.py, benchmarks/compare.py, benchmarks/results/ (3 new files), README.md, PROGRESS.md

**How it was tested**
- pytest: 71 passed on Windows and 3 times in a row on Linux/WSL. New and updated tests:
  - an out-of-order test: the first group is slow to read, so later groups are written first and group 0 last, and the cache is identical to a sequential build with all chunks recorded
  - the interrupt test is now independent of completion order: exactly the decoded chunks are recorded, and the resume decodes only the missing ones
- Byte-identical: the original implementation (before parallel reads) and the new one gave byte-identical images.npy files for RGB/grayscale/RGBA/palette/16-bit PNGs in both output modes.
- Linux smoke runs: baseline --workers 4 saved as imagefolder-baseline-4workers; the Efficio --no-resume-test run leaves the resume fields empty; compare.py groups both baselines.

**Still needs a GPU/Kaggle test**
- Cache build speed with the out-of-order writes on cold NIH files (io_threads 32 and 64).
- The image-folder baseline with --workers 4 for a fair comparison.
- clip_grad_norm values with the CUDA fp16 scaler; the ImageCache report tip on a real run; everything on Colab.

**Open questions**
- Repeat each image-folder run 3 times before quoting a ratio, as for CIFAR.

## 2026-10-22 (NIH Chest X-rays, second Kaggle session: out-of-order build, 4-worker baseline)

**Kaggle run (commit fa066e0, Tesla T4, new session, so the NIH files were cold)**
Two non-overlapping 5,000-image samples from the same seed-0 shuffle: A (rows 0-4,999, the same sample as the first session) and B (rows 5,000-9,999).
- **Cache build, cold files, sample A, io_threads=32: 139 images/s** (train: 4,500 images in 32.46 s = 138.6 images/s, decode capacity 170.5/s; val: 500 in 3.59 s = 139.4/s; max in flight 128). The first session, before the out-of-order writes, built the same sample at 48 images/s; before parallel reads it was 2 images/s. Reads are now close to the decode capacity, so the storage tip correctly stayed quiet; the 4 CPU cores decoding 1024x1024 PNGs are now the limit.
- **Cache build, cold files, sample B, io_threads=64: 148 images/s** (train: 4,500 images in 30.36 s = 148.2/s, decode capacity 161.4/s; val: 500 in 3.87 s = 129.1/s; max in flight 256; values read from the notebook screenshot). That is only about 7% more than with 32 threads because decoding now sets the pace, so the default of 32 stays.
- At about 140 images/s, a cache of all 112,120 NIH images would take about 13 minutes (extrapolated, not measured), against about 39 minutes at 48/s and about 15.6 hours at 2/s.
- **Baseline with 4 workers** (warm files, like the 2-worker run): 34.94 s per epoch, 103.0 epochs/hour, GPU utilization 37.2%, data-wait 74.4%, AUROC 0.615. With 2 workers (first session) it was 54.65 s per epoch: 4 workers make the baseline 1.56x faster, but it still waits for the CPU most of the time.
- **Efficio rerun** (cache reused, --no-resume-test): 5.76 s per epoch, 625.6 epochs/hour, GPU utilization 77.8%, data-wait 0.3%, AUROC 0.673. The resume fields are now empty, as intended, and num_workers is recorded. The first session measured 5.09-5.13 s, so epoch times vary by about 10-15% between Kaggle sessions.
- **Headline comparison, same session:** 625.6 vs 103.0 epochs/hour = 6.07x against the 4-worker baseline. compare.py also prints 9.50x against the 2-worker baseline, but that baseline came from the first session, so that ratio mixes sessions.
- AUROC across all NIH runs: 0.615-0.673 for both pipelines. Single 4-epoch runs on 4,500 images can't show an accuracy difference; that needs longer runs.

**What changed**
- Saved benchmarks/results/imagefolder-baseline-4workers_kaggle_20261022_154332.json and imagefolder-efficio_kaggle_20261022_154436.json, rebuilt from the printed JSON. The cache-build numbers above have no result file (cells 3-4 called ImageCache.build directly) and are recorded only here.
- README: the NIH section now leads with the same-session 4-worker comparison (6.07x), keeps the 2-worker numbers as a note, and keeps the caveats (warm file cache, single runs, AUROC needs longer runs). The io_threads paragraph has the measured build rates.
- Decided not to use Image.reduce before resizing: 13 minutes for the full dataset is acceptable, and it could change the cache output.

**Files touched**
benchmarks/results/ (2 new files), README.md, PROGRESS.md

**How it was tested**
- compare.py on the saved files reproduces the Kaggle comparison: 6.07x against the 4-worker baseline, 9.50x against the 2-worker one.

**Still needs a GPU test**
- clip_grad_norm values with the CUDA fp16 scaler; the ImageCache report tip on a real run; the large-image features on Colab.
- 3 repeats of each image-folder run before quoting a ratio in a release.
