"""Keep a small dataset on the GPU and build batches there. Phase 2 (Person B).

On Colab and Kaggle the GPU often waits for 2-4 CPU cores to decode and augment images.
For datasets that fit in GPU memory (CIFAR, MNIST, FashionMNIST) GPULoader stores the
whole dataset on the GPU once and yields batches with no CPU work per batch.

    from efficio import GPULoader
    from efficio.speed import gpu_augment as A
    loader = GPULoader.from_torchvision(train_set, batch_size=128, transform=A.Compose(
        [A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(), A.Normalize(mean, std)]))

Images are stored as uint8 and turned into floats in [0, 1] one batch at a time, then
`transform` runs on the float batch.
"""

import math
import warnings

import torch

MAX_FREE_FRACTION = 0.3     # refuse datasets that need more than this share of free GPU memory


def _free_gpu_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    return free


def _check_memory(nbytes, device):
    if device.type != "cuda":
        return
    free = _free_gpu_bytes(device)
    if nbytes > MAX_FREE_FRACTION * free:
        raise ValueError(
            f"GPULoader: the dataset needs {nbytes / 1024**3:.2f} GB, more than "
            f"{MAX_FREE_FRACTION:.0%} of the {free / 1024**3:.2f} GB free on {device}. "
            "Use a normal torch DataLoader for this dataset; Efficio will tune it for you.")


def _to_nchw(x):
    """Images as (N, C, H, W). 3-D input (N, H, W) gets one channel. 4-D input is read as
    (N, H, W, C) when the last dim looks like channels (1, 3 or 4) and dim 1 does not."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.shape[3] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        return x.permute(0, 3, 1, 2)
    return x


def _tensor(a):
    return a if torch.is_tensor(a) else torch.as_tensor(a)


def data_nbytes(data):
    """Size of a dataset's `.data` (tensor or numpy array), or None if unknown."""
    if torch.is_tensor(data):
        return data.numel() * data.element_size()
    return getattr(data, "nbytes", None)


def fits_on_gpu(dataset, device=None):
    """True if `dataset` has .data/.targets and would pass GPULoader's memory check."""
    if not torch.cuda.is_available() or not hasattr(dataset, "targets"):
        return False
    nbytes = data_nbytes(getattr(dataset, "data", None))
    if nbytes is None:
        return False
    device = torch.device(device or "cuda")
    return nbytes <= MAX_FREE_FRACTION * _free_gpu_bytes(device)


class GPULoader:
    """Yields (x, y) batches from tensors kept on one device. Use the from_tensors or
    from_torchvision constructors."""

    def __init__(self, x, y, batch_size, shuffle=True, drop_last=False, transform=None,
                 seed=0, device=None):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        x, y = _tensor(x), _tensor(y)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError(f"x and y need the same, non-zero length (got {len(x)} and {len(y)})")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.is_image = x.dim() in (3, 4) and x.dtype == torch.uint8
        if self.is_image:
            x = _to_nchw(x)
        _check_memory(data_nbytes(x) + data_nbytes(y), self.device)
        self.x = x.to(self.device).contiguous()
        self.y = y.to(self.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.seed = seed
        self.epoch = 0
        self._start_batch = 0

    @classmethod
    def from_tensors(cls, x, y, batch_size, shuffle=True, drop_last=False, transform=None,
                     seed=0, device=None):
        """x: uint8 images (N,H,W,C), (N,C,H,W) or (N,H,W), or any other tensor/array of
        N samples. y: N labels."""
        return cls(x, y, batch_size, shuffle, drop_last, transform, seed, device)

    @classmethod
    def from_torchvision(cls, dataset, batch_size, shuffle=True, drop_last=False,
                         transform=None, seed=0, device=None):
        """Uses dataset.data and dataset.targets (CIFAR10/100, MNIST, FashionMNIST).
        The dataset's own CPU transform is not used; pass the GPU version as `transform`."""
        if not (hasattr(dataset, "data") and hasattr(dataset, "targets")):
            raise TypeError(
                f"GPULoader.from_torchvision needs a dataset with .data and .targets "
                f"(CIFAR, MNIST, FashionMNIST); {type(dataset).__name__} has no such "
                "attributes. Load the images and labels into tensors and use "
                "GPULoader.from_tensors(x, y, batch_size) instead.")
        if transform is None and getattr(dataset, "transform", None) is not None:
            warnings.warn("GPULoader ignores the dataset's own transform. Pass the GPU "
                          "equivalent as transform= (see efficio.speed.gpu_augment).")
        return cls(dataset.data, dataset.targets, batch_size, shuffle, drop_last, transform,
                   seed, device)

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def set_epoch(self, epoch):
        """Pick the shuffle order for an epoch. Run.loop calls this; without it, each
        new iteration moves on to the next epoch."""
        self.epoch = epoch

    def set_start_batch(self, n):
        """Start the next iteration at batch n without generating the earlier batches."""
        self._start_batch = n

    def permutation(self, epoch):
        """Sample order for an epoch: the same for the same seed and epoch, on any device."""
        n = len(self.x)
        if not self.shuffle:
            return torch.arange(n, device=self.device)
        gen = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(n, generator=gen).to(self.device)

    def __iter__(self):
        order = self.permutation(self.epoch)
        start, self._start_batch = self._start_batch, 0
        self.epoch += 1
        for b in range(start, len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            x = self.x[idx]
            if self.is_image:
                x = x.float().div_(255)
            if self.transform is not None:
                x = self.transform(x)
            yield x, self.y[idx]
