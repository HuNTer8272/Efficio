# Contributing to Efficio

Thanks for helping. This page covers what Efficio is for, how the repo is laid out, and the rules every change follows.

## Project scope (v1)

Efficio helps people get more training out of every GPU hour on Google Colab, Kaggle and their own GPUs. The target users are students and individuals on free tiers and personal GPUs.

- On Colab and Kaggle, Efficio makes training faster, wastes less quota and saves progress before the limit hits, so a run can continue after the reset or on the other platform.
- On local GPUs there is no quota. There the benefit is speed, crash safety and help with small-memory GPUs. Never describe Efficio as extending time limits on a local GPU.

Out of scope for v1: company cloud training, AWS/GCP, spot instances, multi-node training and FSDP/DeepSpeed tuning. Please don't open pull requests for these.

Efficio supports PyTorch today. TensorFlow/Keras support is planned. Keep `efficio/interfaces.py`, `efficio/env/detect.py` and the budget and report code free of `torch` imports so that support can reuse them. Torch-specific code belongs in `checkpoint/`, `run.py`, `speed/` and `data/`.

## Repo layout

| Path | What it holds |
|---|---|
| `efficio/interfaces.py` | Shared data objects: `HardwareInfo`, `Budget`, `TrainState` |
| `efficio/run.py` | The `Run` class that ties everything together |
| `efficio/env/` | Platform detection (Colab, Kaggle, local) and GPU detection |
| `efficio/checkpoint/` | Atomic checkpoint save and exact resume |
| `efficio/speed/` | DataLoader tuning, channels_last, GPU data pipeline, GPU augmentation |
| `efficio/data/` | Image cache for large image datasets |
| `efficio/report/` | Profiler and end-of-run report |
| `efficio/budget/`, `efficio/adapters/`, `efficio/preflight.py` | Planned features (not usable yet) |
| `examples/` | Small runnable examples |
| `benchmarks/` | Baseline vs Efficio comparisons and their results |
| `tests/` | pytest tests (CPU only, so they run in CI) |

## Install and run the tests

```bash
git clone https://github.com/HuNTer8272/Efficio.git
cd Efficio
pip install -e ".[dev]"
pytest -q
```

The tests run on CPU. Features that need a GPU must also be checked on Colab or Kaggle before they are marked Available in the README.

## Branches and pull requests

1. Branch from an up-to-date `main`: `feat/<short-name>`, `fix/<short-name>` or `release/<version>`.
2. Keep each pull request focused on one change.
3. Before opening a pull request:
   - `pytest -q` passes.
   - New features have tests. Changed behavior has updated tests.
   - README feature statuses are accurate. A feature is **Available** only after it has been verified on a GPU; until then it is marked **Available, needs GPU verification**.
   - User-facing changes are listed in `CHANGELOG.md`.
4. Another maintainer reviews the pull request before it is merged into `main`.
5. Don't force-push `main`. Ask the other maintainer first if history ever has to be rewritten.

## Commit rules

- Commits are authored by the people who wrote the change, using their own name and email in `git config`.
- Write messages that describe the change: a short summary line, then details if needed.
- Credit only human authors. No co-author trailers or "generated with" lines that credit software tools, in commits, pull request titles, pull request descriptions or comments.
- Never commit `.venv/`, `data/`, `efficio_checkpoints/`, caches, build output or secrets. Never put tokens or passwords in code, notebooks or configs.

## Shared interfaces

`efficio/interfaces.py` is the contract between the maintainers. Change it only when both maintainers have agreed, and make the change together in one pull request.

## Benchmarks

- Baseline scripts (`benchmarks/*_baseline.py`) stay plain PyTorch with no Efficio features. Never "optimize" a baseline. If a setting must change for fairness (for example where the data is stored), change it the same way in both scripts.
- Never invent or estimate benchmark numbers. Only report measured results, commit the result JSON files from `benchmarks/results/`, and state the limits of each result (single runs, warm caches and so on).

## Cloning in Colab or Kaggle notebooks

If you clone the repo in a notebook with a token in the URL, remove the token immediately after cloning:

```bash
git clone https://<token>@github.com/HuNTer8272/Efficio.git
cd Efficio
git remote set-url origin https://github.com/HuNTer8272/Efficio.git
```

Kaggle publishes `/kaggle/working` when you save a version, so a token left in `.git/config` would be published. Before sharing a notebook or its output, search it for `github_pat_` and `ghp_`.
