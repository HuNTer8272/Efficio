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

## Checkpoint sizes, save time and resume quality

`checkpoint_sizes.py` compares the exact, light and smallest checkpoint modes (plus trainable-only for the frozen backbone) for three setups:

- ResNet-18 (CIFAR stem) with SGD and momentum
- a small transformer encoder with AdamW
- a frozen ResNet-18 backbone with a small trainable head

Each is trained for 20 steps on synthetic data whose labels the model can learn. For every mode it reports:

- the file size
- the time a synchronous save blocks training
- with background saving: the time training is blocked (the copy to CPU memory) and the time written in the background
- resume quality against an uninterrupted run of the same steps:
  - the max weight difference right after loading
  - the loss and accuracy after one resume and 20 more steps
  - the same after three consecutive save, load and train cycles

A "noise baseline" row shows the same differences for an exact resume with 1e-6 relative noise on the weights. Training is sensitive, so differences of that size are normal variation, not damage from a checkpoint mode.

`verify_light_resume.py` stops a ResNet-18 run with the time budget in the middle of an epoch and resumes it with a fresh `Run`, in light and in smallest mode. It prints PASS or FAIL for: the stop, the checkpoint mode, the resume, exact weights after loading (light) or fp16 rounding at most (smallest), the final step, and the final loss and accuracy against an uninterrupted run.

```bash
python benchmarks/checkpoint_sizes.py            # GPU if available
python benchmarks/checkpoint_sizes.py --quick    # smaller models, a CPU smoke test
python benchmarks/verify_light_resume.py         # add --quick for a CPU smoke test
```

Results of `checkpoint_sizes.py` are saved as `checkpoint_sizes_{platform}_{timestamp}.json`.

## Resume on another platform through storage

`verify_cloud_resume.py` checks that a run started on one machine can finish on another through checkpoint storage. It runs in two phases.

**Phase start** (for example on Kaggle):

- trains ResNet-18 on synthetic data with `resume=False`
- the time budget stops it in the middle of epoch 2; `--budget 11m` uses a real time budget instead
- checks that the stop checkpoint's upload was confirmed before training returned and that it is the manifest's latest
- stores the run settings next to the checkpoints

**Phase finish** (for example on Colab):

- starts from an empty local folder, so the checkpoint must be downloaded
- resumes and finishes the run
- checks that it resumed from the stored checkpoint at the stop step and that the sha256 matches the manifest
- checks that it ends at the same final step as an uninterrupted run
- compares the final loss and accuracy with an uninterrupted run on the same machine. The gaps may be at most 3x those of a noise baseline: the same run stopped and resumed locally with 1e-6 relative weight noise. The floors are 0.05 loss and 3 accuracy points.

On Colab, run it with `%run` in a notebook cell, because Colab secrets (`HF_TOKEN`) are only available inside the notebook kernel.

```bash
python benchmarks/verify_cloud_resume.py --phase start  --storage hf://USER/efficio-verify
python benchmarks/verify_cloud_resume.py --phase finish --storage hf://USER/efficio-verify
python benchmarks/verify_cloud_resume.py --phase both --storage ./verify_store --quick  # CPU
```

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
