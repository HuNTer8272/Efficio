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
