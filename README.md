# Efficio

**Efficio - Get more training out of every GPU hour.**

Free GPUs on Colab and Kaggle come with strict time limits, and much of that time is wasted: idle sessions, slow data loading, full-precision math, and runs that are lost when a session dies. Efficio helps you waste less of it. It is a Python library you add to your training code with a few lines. PyTorch is supported today, with TensorFlow/Keras planned.

## What it does for you

**On Colab and Kaggle**

- Faster training, so each session gets more work done.
- Less wasted quota, so your limit lasts longer.
- Progress saved before the limit hits, so you can continue when the limit resets or on the other platform.

**On your own GPU**

- Faster training.
- Crash safety: your progress survives power cuts, restarts and out-of-memory crashes.
- Help with small-memory GPUs.

Local GPUs have no quota, so there the benefit is speed and safety, not extra time.

## Features

Each item is marked **Available** (in the library and verified on a GPU), **Available, needs GPU verification** (in the library and tested on CPU, not yet verified on a GPU) or **Planned** (on the roadmap).

### Speed

| Feature | Status |
|---|---|
| GPU-aware precision: fp16 on T4, bf16 on newer GPUs | Available |
| Automatic DataLoader tuning: more workers, pinned memory, persistent workers, prefetching | Available |
| channels_last memory format for convolutional models | Available |
| GPU data pipeline for small datasets (GPULoader) | Available |
| Auto batch size with out-of-memory recovery | Planned |
| Preprocessing cache | Planned |
| Use both Kaggle T4 GPUs | Planned |

### Quota stretching

| Feature | Status |
|---|---|
| Stop cleanly before the time budget runs out | Available |
| CPU pre-flight check | Planned |
| Budget planner that fits epochs to your time | Planned |
| Auto-shutdown when training ends | Planned |
| Cheaper validation during training | Planned |

### Smart checkpointing

| Feature | Status |
|---|---|
| Saves model, optimizer, scheduler, gradient scaler, RNG state and exact position | Available |
| Atomic saves, so a crash mid-save never corrupts the last checkpoint | Available |
| Periodic and emergency saves | Available |
| Exact mid-epoch resume: same step count, and the same data order with GPULoader or unshuffled data | Available |
| Same data order after resume with a shuffled DataLoader (per-epoch seeded shuffle) | Available, needs GPU verification |
| Light checkpoint mode, about 3x smaller | Planned |

### Works everywhere

| Feature | Status |
|---|---|
| Detects Colab, Kaggle or local | Available |
| Plain PyTorch support | Available |
| Hugging Face Hub and Google Drive storage for moving runs between platforms | Planned |
| Hugging Face Trainer and PyTorch Lightning support | Planned |
| TensorFlow / Keras support (Keras callback for model.fit) | Planned |

### Visibility

| Feature | Status |
|---|---|
| Basic end-of-run report | Available |
| Profiler: data-wait time and GPU utilization | Available |
| Improved report: setup vs training time, data-wait, GPU utilization, applied optimizations and tips | Available |
| Detailed report with GPU usage, time breakdown, wasted quota and tips | Planned |
| Telegram and email notifications | Planned |

## Quick start

```python
import efficio

run = efficio.Run(budget="2h")            # stop and save 10 minutes before the budget ends
model, optimizer = run.prepare(model, optimizer)   # resumes automatically if a checkpoint exists

for batch in run.loop(loader, epochs=20):
    with run.autocast():                  # fp16 on T4, bf16 on newer GPUs, fp32 on CPU
        loss = compute_loss(model, batch)
    run.backward(loss)
    run.step()

run.report()
```

### Keep a small dataset on the GPU

On Colab the GPU often waits for 2 CPU cores to augment images. If your dataset fits in GPU memory (CIFAR, MNIST, FashionMNIST), `GPULoader` keeps it on the GPU and does the augmentation there, so no CPU work is done per batch. Use it in place of the DataLoader:

```python
from efficio import GPULoader
from efficio.speed import gpu_augment as A

loader = GPULoader.from_torchvision(train_set, batch_size=128, shuffle=True, transform=A.Compose(
    [A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(), A.Normalize(mean, std)]))
for x, y in run.loop(loader, epochs=20):
    ...
```

`from_torchvision` reads `train_set.data` and `train_set.targets` and ignores the dataset's own CPU transform, so pass the GPU version as `transform`. For other data, use `GPULoader.from_tensors(x, y, batch_size)`. GPULoader refuses datasets that would need more than 30% of free GPU memory; use a normal DataLoader for those. The shuffle order depends only on `seed` and the epoch, so a resumed run continues with exactly the same batches.

### Try the example

`examples/quickstart.py` trains a small model on random data. It runs on CPU, Colab or Kaggle.

```bash
python examples/quickstart.py --budget 20s   # stops early and saves a checkpoint
python examples/quickstart.py                # resumes from where the first run stopped
```

## Tested on

Phase 1 has been verified on:

| Platform | GPU | Precision |
|---|---|---|
| Colab | 1x Tesla T4 | fp16 |
| Kaggle | 2x Tesla T4 | fp16 |

On both platforms we checked environment detection, mixed precision, budget stop and exact resume. After resuming mid-epoch, both platforms ended at the identical step count.

Note: on Kaggle, Efficio currently trains on one of the two GPUs.

The Phase 2 speed and visibility features (DataLoader tuning, channels_last, GPULoader, profiler and the improved report) have been verified on Colab (1x Tesla T4) and Kaggle (2x Tesla T4, one used), both in fp16. We checked the tests, budget stop and resume, and the ResNet-18 benchmark below on both platforms, plus GPULoader resume and the SIGTERM save on Colab.

## Measuring results

We will benchmark three workloads, each with and without Efficio:

- ResNet-18 on CIFAR-10
- A DistilBERT fine-tune
- A small LoRA run

We report epochs per quota hour. The numbers below are single 5-epoch runs; before the release they will be averaged over 3 runs.

### ResNet-18 on CIFAR-10, Colab, 1x Tesla T4

5 epochs, batch size 128. Averages exclude the first (warm-up) epoch. Raw results are in `benchmarks/results/`.

| Run | Avg epoch | Epochs per hour | GPU utilization | Test accuracy | Speedup |
|---|---|---|---|---|---|
| Plain PyTorch (fp32) | 47.6 s | 75.6 | 98% | 85.37% | 1.00x |
| Efficio (fp16, DataLoader tuning, channels_last) | 20.7 s | 173.6 | 54% | 85.57% | 2.30x |
| Efficio + GPULoader (`--gpu-data`) | 13.3 s | 270.4 | 96% | 85.59% | 3.58x |

With the normal DataLoader, the GPU waited for Colab's 2 CPU cores to augment images (22% data-wait). GPULoader keeps the data and augmentation on the GPU and brings utilization back to 96%.

### ResNet-18 on CIFAR-10, Kaggle, Tesla T4 (one of the two GPUs)

Same settings as on Colab.

| Run | Avg epoch | Epochs per hour | GPU utilization | Test accuracy | Speedup |
|---|---|---|---|---|---|
| Plain PyTorch (fp32) | 44.8 s | 80.3 | 98% | 85.98% | 1.00x |
| Efficio (fp16, DataLoader tuning, channels_last) | 14.7 s | 244.3 | 89% | 85.55% | 3.04x |
| Efficio + GPULoader (`--gpu-data`) | 14.7 s | 244.3 | 91% | 85.52% | 3.04x |

Kaggle gives the notebook 4 CPU cores, so the tuned DataLoader already keeps the GPU busy (2% data-wait) and GPULoader adds no speed there. GPULoader helps when the CPU is the bottleneck, as on Colab. The report only suggests it in that case.

## Roadmap

- **Phase 1:** done and verified.
- **Phase 2:** in progress.
  - Done so far: profiler, DataLoader tuning, channels_last, GPULoader, improved report.
  - Remaining: image/data cache, light checkpoints, Hugging Face Hub storage, auto batch size, pre-flight check.
- **Phase 3:** Google Drive storage, auto-shutdown, budget planner, full report.
- **Phase 4:** multi-GPU, notifications, Hugging Face Trainer, PyTorch Lightning and TensorFlow/Keras support.
- **Phase 5:** benchmarks, docs, PyPI release.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`efficio/interfaces.py` holds the shared data objects. Only change it when both teammates agree, and make the change together.
