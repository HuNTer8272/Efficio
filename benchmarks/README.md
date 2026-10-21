# Benchmarks

These scripts measure how much more training Efficio gets out of a GPU hour. Each workload is trained twice, once with plain PyTorch and once through Efficio. Both runs use the same model, data, batch size, seed and learning-rate schedule.

## ResNet-18 on CIFAR-10

| Script | What it does |
|---|---|
| `resnet_cifar_baseline.py` | Plain PyTorch in fp32, with no Efficio and no optimizations. This is the reference run. Do not add speed-ups to it. |
| `resnet_cifar_efficio.py` | The same training, run through `efficio.Run` (mixed precision chosen for the GPU, checkpointing on). |
| `common.py` | Shared model, data, optimizer, timing and result saving. It does not import Efficio. |
| `compare.py` | Prints all saved results as a table, plus the Efficio speedup per platform. |

The model is torchvision's ResNet-18 with a CIFAR stem (3x3 first conv, no max-pool). It trains with SGD (lr 0.1, momentum 0.9, weight decay 5e-4) and a OneCycle schedule. CIFAR-10 is stored in `./data` (change it with `--data-dir`), which git ignores. Before downloading, `common.py` looks for an attached copy under `/kaggle/input` (a `cifar-10-batches-py` folder or `cifar-10-python.tar.gz`) and copies or extracts it, because the download can be very slow on Kaggle. It prints which source it used. On Kaggle, attach a CIFAR-10 dataset to the notebook to skip the download.

## What is measured

Each run saves these numbers:

- time per epoch, and the average epoch time without the first (warm-up) epoch
- epochs per hour (the headline number)
- images per second
- peak GPU memory
- average GPU utilization (sampled with `nvidia-smi` every 0.5 s)
- test accuracy

In the Efficio run, the last epoch's time also includes the final checkpoint save.

## How to run

Run these from the repo root on a GPU machine (Colab, Kaggle or local):

```bash
pip install -e .
python benchmarks/resnet_cifar_baseline.py --epochs 5
python benchmarks/resnet_cifar_efficio.py --epochs 5
python benchmarks/compare.py
```

Both scripts also accept `--batch-size` (default 128) and `--data-dir` (default `./data`). Use the same batch size for both.

To measure the DataLoader tuning and channels_last separately, run the Efficio script with `--no-optimize-data` as well. That run keeps mixed precision but turns off those two optimizations, and is saved as `efficio-no-data-opt`:

```bash
python benchmarks/resnet_cifar_efficio.py --epochs 5 --no-optimize-data
```

To measure the GPU data pipeline, add `--gpu-data`. The training set is then kept on the GPU with `GPULoader` and augmented there with the same augmentation as the baseline (random crop 32 with padding 4, horizontal flip, the same normalization), the same batch size and `drop_last=True`. The test set still uses the normal loader. The run is saved as `efficio-gpu-data`, or `efficio-gpu-data-no-data-opt` together with `--no-optimize-data`. GPULoader shuffles with its own seeded order, so the sample order differs from the baseline's DataLoader.

```bash
python benchmarks/resnet_cifar_efficio.py --epochs 5 --gpu-data
```

## Results

Each run writes a JSON file to `benchmarks/results/`, named `{run}_{platform}_{timestamp}.json`. Commit these files so the whole team can compare results across platforms. Only report numbers that came from a real run.

## Planned next

- A DistilBERT fine-tune
- A small LoRA run
