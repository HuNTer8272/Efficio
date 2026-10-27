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

## Image folder (large images decoded from disk)

This benchmark measures the case GPULoader can't handle: images that don't fit in GPU memory and are decoded from files every epoch, as with a 100k-image X-ray dataset.

| Script | What it does |
|---|---|
| `image_folder_baseline.py` | Plain PyTorch in fp32. A Dataset opens, decodes, resizes and augments every PNG on the CPU in each epoch (torchvision transforms, 2 DataLoader workers). Do not add speed-ups to it. |
| `image_folder_efficio.py` | Builds an ImageCache once (timed and saved as `cache_build_s`; reused in later runs), then reads uint8 arrays and runs the same augmentation on the GPU through `gpu_transform`, with mixed precision and the DataLoader tuning. |
| `image_common.py` | Shared data, split, model (ResNet-18, multi-label head), image size (224), batch size (64), augmentation settings and seed. It does not import Efficio. |

Both use the same augmentation: resize to 224, random resized crop (scale 0.8-1.0), small random affine (7 degrees, 5% shift, 0.95-1.05 scale), horizontal flip, brightness/contrast 0.1, grayscale to 3 channels, ImageNet normalization.

The Efficio run also tests budget stop and resume. After 2 full epochs, the time budget is set to run out halfway through epoch 3. The run saves and stops, then a fresh `Run` resumes and finishes. The result file records where it stopped, where it resumed and whether the final step count matches (`resume.ok`). The split epoch is left out of the epoch-time average.

Data sources:

```bash
# Generated data: 5,000 grayscale 512x512 PNGs (written once to ./data/image_folder)
python benchmarks/image_folder_baseline.py --fake --epochs 6
python benchmarks/image_folder_efficio.py --fake --epochs 6

# A real dataset: an image folder plus a CSV (filename + one 0/1 column per label)
python benchmarks/image_folder_baseline.py --images /kaggle/input/xrays/images --labels /kaggle/input/xrays/labels.csv --limit 10000
python benchmarks/image_folder_efficio.py --images /kaggle/input/xrays/images --labels /kaggle/input/xrays/labels.csv --limit 10000
python benchmarks/compare.py
```

Results are saved as `imagefolder-baseline` and `imagefolder-efficio` with `"benchmark": "image-folder"`, and `compare.py` computes speedups per benchmark. The fake labels are random, so accuracy and AUROC on `--fake` data only confirm that training runs; compare speed there, not accuracy. `test_acc_pct` is the mean per-label accuracy at threshold 0.5, and `val_auroc` is the mean ROC AUC.

## Final GPU check before a release

`verify_gpu_features.py` prints PASS, FAIL or SKIP for the features that need a GPU to verify: `run.clip_grad_norm` with the real fp16 GradScaler (with and without gradient accumulation), the ImageCache report tip for image files decoded every epoch, and a budget stop with resume. On CPU the CUDA-only checks print SKIP. It uses the installed `efficio` package.

```bash
python benchmarks/verify_gpu_features.py
```

## Results

Each run writes a JSON file to `benchmarks/results/`, named `{run}_{platform}_{timestamp}.json`. Commit these files so the whole team can compare results across platforms. Only report numbers that came from a real run.

## Planned next

- A DistilBERT fine-tune
- A small LoRA run
