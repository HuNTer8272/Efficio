"""Shared setup for the ResNet-18 / CIFAR-10 benchmark.

Both the baseline and the Efficio run import from here, so they use the identical
model, data, batch size and LR schedule. This file must not import efficio.
"""

import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch import nn

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

KAGGLE_INPUT = Path("/kaggle/input")
CIFAR_FOLDER = "cifar-10-batches-py"
CIFAR_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR_FILES = ["batches.meta", "test_batch"] + [f"data_batch_{i}" for i in range(1, 6)]


def platform_name():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def _find_attached_cifar(root, max_depth=4):
    """Look for an attached CIFAR-10 copy under `root` (Kaggle: /kaggle/input).
    Returns ("folder", path), ("archive", path) or None."""
    root = Path(root)
    if not root.is_dir():
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here.name == CIFAR_FOLDER and all(f in filenames for f in CIFAR_FILES):
            return "folder", here
        if CIFAR_ARCHIVE in filenames:
            return "archive", here / CIFAR_ARCHIVE
        if len(here.relative_to(root).parts) >= max_depth:
            dirnames[:] = []        # don't walk deep into large attached datasets
    return None


def prepare_cifar10(data_dir, search_root=KAGGLE_INPUT):
    """Put CIFAR-10 into data_dir from the fastest source, so torchvision skips its slow
    download (59 min on Kaggle once). Prints and returns which source was used:
    "existing", "folder" (copied), "archive" (extracted) or "download"."""
    data_dir = Path(data_dir)
    target = data_dir / CIFAR_FOLDER
    if all((target / f).exists() for f in CIFAR_FILES):
        print(f"CIFAR-10 source: existing copy in {target}")
        return "existing"
    found = _find_attached_cifar(search_root)
    if found is None:
        print(f"CIFAR-10 source: no copy in {data_dir} or {search_root}, downloading with "
              "torchvision (on Kaggle, attach a CIFAR-10 dataset to skip this)")
        return "download"
    kind, path = found
    start = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    if kind == "folder":
        shutil.copytree(path, target, dirs_exist_ok=True)
        print(f"CIFAR-10 source: copied attached folder {path} to {target}", end="")
    else:
        with tarfile.open(path) as tar:
            try:
                tar.extractall(data_dir, filter="data")
            except TypeError:       # Python without extraction filters
                tar.extractall(data_dir)
        print(f"CIFAR-10 source: extracted attached archive {path} to {data_dir}", end="")
    print(f" ({time.perf_counter() - start:.1f} s)")
    return kind


def make_loaders(batch_size=128, num_workers=2, data_dir="./data"):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)])
    prepare_cifar10(data_dir)
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False,
                                              num_workers=num_workers)
    return train_loader, test_loader


def make_model():
    model = torchvision.models.resnet18(num_classes=10)
    # CIFAR stem: 3x3 conv, no downsampling, no maxpool (images are only 32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_optimizer(model, epochs, steps_per_epoch):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
                                                    total_steps=epochs * steps_per_epoch)
    return optimizer, scheduler


class GpuUtilSampler:
    """Samples GPU 0 utilization with nvidia-smi every 0.5 s in a background thread.
    Does nothing without CUDA. `average` is None if no samples were taken."""

    CMD = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"]

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(self.CMD, capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if torch.cuda.is_available():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return False

    @property
    def average(self):
        return sum(self.samples) / len(self.samples) if self.samples else None


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate(model, loader, device, autocast=None):
    """Test accuracy in %. `autocast` is an optional function returning a context manager."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if autocast is not None:
                with autocast():
                    out = model(x)
            else:
                out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return 100.0 * correct / total


def save_results(run_name, epoch_times, images_per_epoch, avg_util, test_acc, extra):
    platform = platform_name()
    cuda = torch.cuda.is_available()
    # Skip the first epoch (warm-up: cuDNN autotuning, worker start-up, dataset caching)
    timed = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    avg_epoch = sum(timed) / len(timed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "run": run_name,
        "platform": platform,
        "gpu": torch.cuda.get_device_name(0) if cuda else "none",
        "torch_version": torch.__version__,
        "timestamp": stamp,
        "epochs": len(epoch_times),
        "epoch_times_s": [round(t, 3) for t in epoch_times],
        "avg_epoch_s_excl_first": round(avg_epoch, 3),
        "epochs_per_hour": round(3600 / avg_epoch, 3),
        "images_per_s": round(images_per_epoch / avg_epoch, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if cuda else 0.0,
        "avg_gpu_util_pct": round(avg_util, 1) if avg_util is not None else None,
        "test_acc_pct": round(test_acc, 2),
    }
    result.update(extra)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_name}_{platform}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved {path}")
    return path
