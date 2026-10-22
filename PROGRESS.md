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
