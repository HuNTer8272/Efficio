"""Batched image augmentation that runs on the GPU. Phase 2 (Person B).

Every function takes a float batch shaped (N, C, H, W) on any device and draws its
randomness per sample from the global torch RNG of that device. That RNG is saved in
Efficio checkpoints, so augmentation continues exactly after a resume.

The classes are thin wrappers so a pipeline reads like torchvision:
    aug = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), Normalize(mean, std)])
"""

import torch
import torch.nn.functional as F


def _pair(size):
    return (size, size) if isinstance(size, int) else tuple(size)


def random_crop(x, size, padding=0):
    """Zero-pad each side by `padding`, then cut a random size x size window per sample.
    Same as torchvision RandomCrop(size, padding=padding)."""
    th, tw = _pair(size)
    if padding:
        x = F.pad(x, (padding, padding, padding, padding))
    n, _, h, w = x.shape
    if (h, w) == (th, tw):
        return x
    if h < th or w < tw:
        raise ValueError(f"crop size {(th, tw)} is larger than the padded image {(h, w)}")
    top = torch.randint(0, h - th + 1, (n,), device=x.device)
    left = torch.randint(0, w - tw + 1, (n,), device=x.device)
    rows = (top[:, None] + torch.arange(th, device=x.device))[:, :, None]     # (N, th, 1)
    cols = (left[:, None] + torch.arange(tw, device=x.device))[:, None, :]    # (N, 1, tw)
    idx = torch.arange(n, device=x.device)[:, None, None]
    # Advanced indexing around the channel slice puts channels last: (N, th, tw, C)
    return x[idx, :, rows, cols].permute(0, 3, 1, 2).contiguous()


def random_horizontal_flip(x, p=0.5):
    """Mirror each sample left-right with probability p."""
    flip = torch.rand(x.shape[0], device=x.device) < p
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def normalize(x, mean, std):
    """(x - mean) / std per channel."""
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


class Compose:
    """Apply callables in order."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomCrop:
    def __init__(self, size, padding=0):
        self.size, self.padding = size, padding

    def __call__(self, x):
        return random_crop(x, self.size, self.padding)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return random_horizontal_flip(x, self.p)


class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return normalize(x, self.mean, self.std)
