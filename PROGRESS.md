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
