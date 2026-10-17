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

Each item is marked **Available** (in the library today) or **Planned** (on the roadmap).

### Speed

| Feature | Status |
|---|---|
| GPU-aware precision: fp16 on T4, bf16 on newer GPUs | Available |
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
| Exact mid-epoch resume | Available |
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

## Measuring results

We will benchmark three workloads, each with and without Efficio:

- ResNet-18 on CIFAR-10
- A DistilBERT fine-tune
- A small LoRA run

We will report epochs per quota hour. Results are coming soon.

## Roadmap

- **Phase 1:** done and verified.
- **Phase 2:** Hugging Face Hub storage, light checkpoints, auto batch size, pre-flight check.
- **Phase 3:** Google Drive storage, auto-shutdown, data cache, budget planner, full report.
- **Phase 4:** multi-GPU, notifications, Hugging Face Trainer, PyTorch Lightning and TensorFlow/Keras support.
- **Phase 5:** benchmarks, docs, PyPI release.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`efficio/interfaces.py` holds the shared data objects. Only change it when both teammates agree, and make the change together.
