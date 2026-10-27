# Efficio

**Get more training out of every GPU hour on Colab, Kaggle and your own GPU.** Efficio is a small PyTorch library that picks the right mixed precision, keeps the GPU fed with data, stops cleanly before your time runs out and resumes exactly where it stopped.

```bash
pip install efficio
```

## Quick start on Colab

```python
!pip install -q efficio
import efficio, torch

run = efficio.Run(budget="2h")                        # save and stop 10 minutes before 2 h
model, optimizer = run.prepare(model, optimizer)      # resumes from the last checkpoint
for x, y in run.loop(train_loader, epochs=20):        # batches arrive on the GPU
    with run.autocast():                              # fp16 on a T4, bf16 on newer GPUs
        loss = torch.nn.functional.cross_entropy(model(x), y)
    run.backward(loss)
    run.step()
run.report()                                          # time split, GPU use and tips
```

Run the same cell again after a disconnect or in a new session and training continues from the last checkpoint. A complete CIFAR-10 example is in the [Colab quick-start notebook](https://github.com/HuNTer8272/Efficio/blob/main/examples/colab_quickstart.ipynb).

## What it does for you

Free GPUs on Colab and Kaggle have strict time limits, and much of that time is wasted: slow data loading, full-precision math and runs lost when a session dies.

- **On Colab and Kaggle:** faster training, less wasted quota, and progress saved before the limit hits, so you can continue after the reset or on the other platform.
- **On your own GPU:** faster training and crash safety (power cuts, restarts, out-of-memory crashes). Local GPUs have no quota, so there the benefit is speed and safety, not extra time.

## Features

### Speed

- **GPU-aware mixed precision:** fp16 with gradient scaling on a T4 and older GPUs, bf16 on Ampere and newer, fp32 on CPU (`run.autocast()`).
- **Automatic DataLoader tuning:** more workers, pinned memory, persistent workers and prefetching, applied by `run.loop()` on Linux (Colab, Kaggle).
- **channels_last** memory format for convolutional models on the GPU, plus `cudnn.benchmark`.
- **GPULoader:** keeps a small dataset (CIFAR, MNIST, FashionMNIST) on the GPU and builds and augments batches there, with no CPU work per batch.
- **ImageCache:** decodes and resizes a large image dataset once, using all CPU cores and parallel file reads, into one uint8 array that later sessions reuse. The build is resumable.
- **GPU augmentation** for any DataLoader: crop, resized crop, affine, flip, brightness and contrast, normalize (`efficio.gpu_augment`, used through `run.loop(..., gpu_transform=...)`).

### Time budget and checkpoints

- **Budget stop:** `Run(budget="2h")` saves and stops before the time runs out. A shutdown signal (SIGTERM) also triggers a save.
- **Full checkpoints:** model, optimizer, scheduler, gradient scaler, RNG states and the exact position in the epoch.
- **Atomic saves**, so a crash in the middle of a save never corrupts the last good checkpoint. Periodic and emergency saves.
- **Exact mid-epoch resume:** the same step count, and the same data order with GPULoader or unshuffled data.
- **Gradient accumulation** (`Run(grad_accum_steps=...)`) and a validation helper (`run.evaluate_mode()`: eval mode, no gradients, mixed precision).
- **Smaller checkpoints** (`Run(checkpoint_mode="light")` or `"smallest"`): light keeps the weights at full precision and stores the optimizer state in 8 bits, about 2x smaller for Adam-trained models; smallest also stores the weights in fp16, about 3x smaller. Verified on Colab T4. Not in a PyPI release yet.
- **Trainable-only checkpoints** (`Run(save_trainable_only="auto")`): with a frozen backbone, only the trainable parameters are saved, and resuming checks that the frozen weights are the same. Verified on Colab T4. Not in a PyPI release yet.
- **Background checkpoint saving** (`Run(async_save=True)`): periodic saves are written in a background thread; budget, shutdown and final saves always finish first. Verified on Colab T4. Not in a PyPI release yet.
- Resume quality of the checkpoint modes was checked on the T4 against a noise baseline (an exact resume with tiny weight noise); see [Checkpoint options](#checkpoint-options).

### Visibility

- **End-of-run report:** setup vs training time, time spent waiting for data, average GPU utilization, GPU minutes used, the optimizations applied, and plain-language tips (for example: use GPULoader, use ImageCache, or try a larger batch).

### Available, needs GPU verification

These work and are covered by CPU tests, but haven't been confirmed on a GPU yet:

- **Same data order after a resume with a shuffled DataLoader** (per-epoch seeded shuffle).
- **Gradient clipping that works with fp16 and gradient accumulation** (`run.clip_grad_norm(max_norm)`). It has run on a Kaggle T4 without errors; its values with the fp16 scaler are only checked on CPU so far.
- **Report tip for image datasets decoded from files every epoch** (suggests ImageCache).

The large-image features (ImageCache, CachedImageDataset, GPU augmentation through `gpu_transform`, `evaluate_mode`) have been verified on Kaggle (Tesla T4) but not yet on Colab.

## Usage

### Keep a small dataset on the GPU

```python
from efficio import GPULoader, gpu_augment as A

loader = GPULoader.from_torchvision(train_set, batch_size=128, shuffle=True, transform=A.Compose(
    [A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(), A.Normalize(mean, std)]))
for x, y in run.loop(loader, epochs=20):
    ...
```

`from_torchvision` reads `train_set.data` and `train_set.targets` and ignores the dataset's own CPU transform, so pass the GPU version as `transform`. For other data use `GPULoader.from_tensors(x, y, batch_size)`. GPULoader refuses datasets that would need more than 30% of free GPU memory; use a normal DataLoader for those.

### Large image datasets

```python
from torch.utils.data import DataLoader
import efficio
from efficio import CachedImageDataset, ImageCache, gpu_augment as A

ImageCache.build(paths, labels, "/content/cache/train", size=(224, 224), grayscale=True)
loader = DataLoader(CachedImageDataset("/content/cache/train"), batch_size=64, shuffle=True,
                    num_workers=2)
augment = A.Compose([A.ToFloat(), A.RepeatChannels(3), A.RandomResizedCrop(224, scale=(0.8, 1.0)),
                     A.RandomHorizontalFlip(), A.Normalize(mean, std)])
for x, y in run.loop(loader, epochs=10, gpu_transform=efficio.on_inputs(augment)):
    ...
```

`labels` can be class ids or a multi-hot array. Building the cache needs no GPU: to save GPU quota, build it in a CPU-only session, keep the folder (Google Drive on Colab; notebook output or a private dataset on Kaggle) and train in a GPU session. With `workers > 1` on Windows or macOS, call `ImageCache.build` under `if __name__ == "__main__":`.

### Gradient clipping and validation

```python
run.backward(loss)
run.clip_grad_norm(1.0)          # between backward and step; real units, also with fp16
run.step()

with run.evaluate_mode():        # eval mode, no gradients, mixed precision
    val_loss = loss_fn(model(x_val), y_val)
```

### Checkpoint options

```python
run = efficio.Run(
    budget="2h",
    checkpoint_mode="exact",        # or "light" (~2x smaller) or "smallest" (~3x smaller)
    save_trainable_only="auto",     # "auto": skip frozen parameters when some are frozen
    async_save=True,                # periodic saves written in the background
)
```

Three checkpoint modes. Any mode can resume from a checkpoint written in any other mode.

| Mode | What is stored | Size (Adam) | Resume | Use it when |
|---|---|---|---|---|
| `"exact"` (default) | everything at full precision | 1x | bit-identical | you want the safest choice |
| `"light"` | full-precision weights, 8-bit optimizer state | about 2x smaller | weights exact; optimizer state approximate, a tiny effect on training | you want smaller files without rounding any weights |
| `"smallest"` | fp16 weights, 8-bit optimizer state | about 3x smaller | weights rounded to fp16, a slight effect after resume | file size matters most (slow uploads, little disk space) |

Checkpoint sizes measured on a Colab T4 (`benchmarks/checkpoint_sizes.py`). These are results for these models, not a guarantee; the saving depends on the model and the optimizer.

| Model | Optimizer | exact | light | smallest | trainable only |
|---|---|---|---|---|---|
| ResNet-18 (CIFAR stem) | SGD + momentum | 85.36 MB | 53.75 MB | 32.42 MB | - |
| Small transformer encoder | AdamW | 59.69 MB | 30.26 MB | 20.32 MB | - |
| Frozen ResNet-18 backbone + small head | AdamW | 44.23 MB | 43.47 MB | 21.90 MB | 1.61 MB |

With a frozen backbone, the optimizer state only covers the head, so light saves little; trainable-only saving is what shrinks those checkpoints.

Resume quality was checked on the same T4 with `benchmarks/verify_light_resume.py`: a ResNet-18 run stopped by the time budget mid-epoch and resumed with a fresh `Run`, 27 of 27 checks passed. Compared with an uninterrupted run:

- **Exact:** bit-identical.
- **Restored state:** in every mode the gradient scaler, scheduler and step count came back exactly, and none of the 11,173,962 second-moment values was zeroed or halved.
- **Final loss gaps:** noise baseline (an exact resume with 1e-6 weight noise) 0.0005, light 0.0012, smallest 0.0001. Accuracy gap 0.0 points everywhere.

- **Light and smallest** compress large optimizer state tensors (for example Adam's `exp_avg` and `exp_avg_sq`, SGD momentum) with 8-bit blockwise quantization. For Adam, `exp_avg_sq` is stored as its square root and `exp_avg` as its ratio to it, so the update direction after a resume matches the saved one within one quantization step. With SGD the saving is smaller (1.6x for light in the table above), because momentum is half of the checkpoint.
- **Smallest** also stores floating-point weights in fp16 and casts them back to their original dtype on load. Integer buffers are unchanged.
- In both modes, the scheduler, gradient scaler, RNG states and position are saved exactly.
- **`save_trainable_only="auto"`** (default) leaves frozen parameters (`requires_grad=False`) out of the checkpoint when the model has any, and saves a signature of them instead (names, shapes, dtypes and a checksum). On resume, the model must have the same frozen weights, so load the same pretrained model and freeze the same layers before `run.prepare()`. If they differ, Efficio raises an error instead of resuming. `True` and `False` force it on or off. Buffers such as BatchNorm statistics are always saved.
- **`async_save=True`** (default) copies the state to CPU memory at a periodic save and writes the file in a background thread, so training continues. Only one save runs at a time. Saves for the time budget, a shutdown signal and the end of training wait for any background save and then write synchronously. If a background save fails, the error is raised at the next `run.step()`. `run.save()` called by you waits until the file is written.

Checkpoints written by Efficio 0.1.x still load and resume. The end-of-run report shows the checkpoint mode, the size of the last file and how much save time was hidden by background saving.

## Results

These are single runs on free notebook GPUs, so read them as examples, not guarantees: speedups depend on the model, the data pipeline and how many CPU cores you get. The baselines are plain PyTorch in fp32 without any Efficio feature. Epoch-time averages leave out the first (warm-up) epoch. The raw results are in [`benchmarks/results/`](https://github.com/HuNTer8272/Efficio/tree/main/benchmarks/results) and the scripts in [`benchmarks/`](https://github.com/HuNTer8272/Efficio/tree/main/benchmarks).

### ResNet-18 on CIFAR-10, Colab, 1x Tesla T4

5 epochs, batch size 128.

| Run | Avg epoch | Epochs per hour | GPU utilization | Test accuracy | Speedup |
|---|---|---|---|---|---|
| Plain PyTorch (fp32) | 47.6 s | 75.6 | 98% | 85.37% | 1.00x |
| Efficio (fp16, DataLoader tuning, channels_last) | 20.7 s | 173.6 | 54% | 85.57% | 2.30x |
| Efficio + GPULoader | 13.3 s | 270.4 | 96% | 85.59% | 3.58x |

With the normal DataLoader the GPU waited for Colab's 2 CPU cores to augment images (22% data-wait). GPULoader keeps the data and augmentation on the GPU.

### ResNet-18 on CIFAR-10, Kaggle, Tesla T4 (one of the two GPUs)

Same settings as on Colab.

| Run | Avg epoch | Epochs per hour | GPU utilization | Test accuracy | Speedup |
|---|---|---|---|---|---|
| Plain PyTorch (fp32) | 44.8 s | 80.3 | 98% | 85.98% | 1.00x |
| Efficio (fp16, DataLoader tuning, channels_last) | 14.7 s | 244.3 | 89% | 85.55% | 3.04x |
| Efficio + GPULoader | 14.7 s | 244.3 | 91% | 85.52% | 3.04x |

Kaggle gives the notebook 4 CPU cores, so the tuned DataLoader already keeps the GPU busy (2% data-wait) and GPULoader adds no speed there.

### Large images from disk: NIH Chest X-rays, Kaggle, Tesla T4

5,000 randomly sampled 1024x1024 PNGs from `/kaggle/input` (4,500 train / 500 validation), 14 findings as a multi-label target, ResNet-18 at 224x224, batch size 64, 4 epochs. The baseline decodes, resizes and augments every PNG on the CPU in each epoch with 4 DataLoader workers; Efficio builds an ImageCache once and augments on the GPU. Both ran in the same Kaggle session.

| Run | Avg epoch | Epochs per hour | GPU utilization | Data-wait | Validation AUROC |
|---|---|---|---|---|---|
| Plain PyTorch (fp32, CPU decoding, 4 workers) | 34.9 s | 103.0 | 37% | 74% | 0.615 |
| Efficio (fp16, ImageCache + GPU augmentation) | 5.8 s | 625.6 | 78% | 0.3% | 0.673 |

Building the cache took about 36 s on cold files (139 images/s); later runs reuse it. Caveats:

- **Warm file cache:** the baseline ran after the cache build, so the PNGs were already in RAM, which makes its file reads faster than on a cold start.
- **Single runs:** each row is one 4-epoch run. Efficio's epoch time varied by about 10-15% between sessions (5.1 to 5.8 s).
- **AUROC, not accuracy:** most labels are 0, so per-label accuracy is about 95% for every run. Across the NIH runs, AUROC ranged from 0.615 to 0.673 for both pipelines, so 4 epochs on 4,500 images can't show an accuracy difference either way.

## Works with

- PyTorch 2.0 or newer, Python 3.10 or newer
- Google Colab and Kaggle notebooks
- A local NVIDIA GPU, or CPU (for development and testing)

## Known limitations

- **PyTorch only.** TensorFlow/Keras support is planned.
- **Single GPU.** On Kaggle's 2x T4 machines, Efficio trains on one GPU.
- **GPULoader** needs a dataset with `.data` and `.targets` (`from_torchvision`), or tensors passed to `GPULoader.from_tensors`.
- **Windows and macOS:** Efficio doesn't add DataLoader workers automatically there, because worker processes need a picklable dataset and an `if __name__ == "__main__":` guard. Set `num_workers` yourself to use them.

## Roadmap

Planned for future releases:

- Hugging Face Hub and Google Drive storage for moving runs between platforms
- Automatic batch size with out-of-memory recovery
- A CPU pre-flight check, a budget planner and auto-shutdown when training ends
- Both Kaggle T4 GPUs
- Hugging Face Trainer, PyTorch Lightning and TensorFlow/Keras support
- Telegram and email notifications
- More benchmarks: a DistilBERT fine-tune and a small LoRA run

## Contributing

See [CONTRIBUTING.md](https://github.com/HuNTer8272/Efficio/blob/main/CONTRIBUTING.md) for the development setup, tests and pull request rules. Changes are listed in the [changelog](https://github.com/HuNTer8272/Efficio/blob/main/CHANGELOG.md).

## License

MIT. See [LICENSE](https://github.com/HuNTer8272/Efficio/blob/main/LICENSE).
