"""Batched image augmentation that runs on the GPU.

Every function takes a batch shaped (N, C, H, W) on any device and draws its randomness
per sample from the global torch RNG of that device. That RNG is saved in Efficio
checkpoints, so augmentation continues exactly after a resume. Except for to_float and
repeat_channels, the functions expect float images; the random geometry and color ops
expect values in [0, 1], so use them before normalize.

The classes are thin wrappers so a pipeline reads like torchvision:
    aug = Compose([ToFloat(), RandomCrop(32, padding=4), RandomHorizontalFlip(),
                   Normalize(mean, std)])
Use with GPULoader(transform=aug), or with any DataLoader through
run.loop(loader, epochs, gpu_transform=on_inputs(aug)).
"""

import math

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


def to_float(x):
    """uint8 images -> float in [0, 1]. Float input is returned unchanged."""
    return x.float().div_(255) if x.dtype == torch.uint8 else x


def repeat_channels(x, n=3):
    """Grayscale (1 channel) -> n identical channels, e.g. for ImageNet-pretrained models."""
    if x.shape[1] == n:
        return x
    if x.shape[1] != 1:
        raise ValueError(f"repeat_channels needs 1 or {n} channels, got {x.shape[1]}")
    return x.expand(-1, n, -1, -1)


def _need_float(x, name):
    if not x.is_floating_point():
        raise TypeError(f"{name} needs float images; put ToFloat() first")


def resize(x, size, antialias=True):
    """Resize every image to `size` (bilinear)."""
    _need_float(x, "resize")
    return F.interpolate(x, size=_pair(size), mode="bilinear", align_corners=False,
                         antialias=antialias)


def _uniform(n, low, high, x):
    return torch.empty(n, device=x.device).uniform_(low, high)


def _warp(x, theta, size):
    """Sample each image through its own 2x3 affine matrix (output -> input coordinates,
    normalized to [-1, 1]). Points outside the image become 0."""
    n, c = x.shape[:2]
    grid = F.affine_grid(theta.to(x.dtype), (n, c, *size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_resized_crop(x, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
    """Cut a random box per sample (area fraction in `scale`, aspect ratio in `ratio`) and
    resize it to `size`, like torchvision RandomResizedCrop. Boxes that would not fit are
    clamped to the image instead of retried."""
    _need_float(x, "random_resized_crop")
    n, _, h, w = x.shape
    area = h * w * _uniform(n, scale[0], scale[1], x)
    r = torch.exp(_uniform(n, math.log(ratio[0]), math.log(ratio[1]), x))
    cw = torch.sqrt(area * r).clamp(max=w)                       # box size in pixels
    ch = torch.sqrt(area / r).clamp(max=h)
    cx = cw / 2 + torch.rand(n, device=x.device) * (w - cw)      # box center in pixels
    cy = ch / 2 + torch.rand(n, device=x.device) * (h - ch)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 2] = cw / w, 2 * cx / w - 1
    theta[:, 1, 1], theta[:, 1, 2] = ch / h, 2 * cy / h - 1
    return _warp(x, theta, _pair(size))


def random_affine(x, degrees=10.0, translate=(0.0, 0.0), scale=(1.0, 1.0)):
    """Rotate by up to +-degrees, shift by up to translate=(fraction of width, of height) and
    scale by a factor in `scale`, independently per sample. Uncovered areas become 0."""
    _need_float(x, "random_affine")
    n, _, h, w = x.shape
    angle = torch.deg2rad(_uniform(n, -degrees, degrees, x))
    s = _uniform(n, scale[0], scale[1], x)
    tx = _uniform(n, -translate[0], translate[0], x) * 2         # in normalized units
    ty = _uniform(n, -translate[1], translate[1], x) * 2
    cos, sin = torch.cos(angle) / s, torch.sin(angle) / s
    # Inverse rotation/scale in pixel space, expressed in normalized coordinates
    a = torch.stack([torch.stack([cos, sin * h / w], 1),
                     torch.stack([-sin * w / h, cos], 1)], 1)    # (N, 2, 2)
    shift = torch.stack([tx, ty], 1).unsqueeze(2)                # (N, 2, 1)
    theta = torch.cat([a, -a @ shift], 2)
    return _warp(x, theta, (h, w))


def random_brightness_contrast(x, brightness=0.2, contrast=0.2):
    """Scale brightness by a factor in [1 - brightness, 1 + brightness] and contrast by one
    in [1 - contrast, 1 + contrast] per sample. Expects values in [0, 1]; clamps to it."""
    _need_float(x, "random_brightness_contrast")
    n = x.shape[0]
    b = _uniform(n, 1 - brightness, 1 + brightness, x).to(x.dtype).view(n, 1, 1, 1)
    c = _uniform(n, 1 - contrast, 1 + contrast, x).to(x.dtype).view(n, 1, 1, 1)
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean).clamp(0, 1)


def on_inputs(transform):
    """Wrap a transform so it only touches the inputs of a batch: the first element of an
    (inputs, labels, ...) tuple/list, or batch["pixel_values"] / batch["image"] of a dict
    batch. A plain tensor batch is transformed as a whole. For run.loop(gpu_transform=...)."""
    def apply(batch):
        if torch.is_tensor(batch):
            return transform(batch)
        if isinstance(batch, dict):
            for key in ("pixel_values", "image"):
                if key in batch:
                    return {**batch, key: transform(batch[key])}
            raise KeyError('on_inputs: dict batch has no "pixel_values" or "image" key')
        if isinstance(batch, (list, tuple)):
            return type(batch)([transform(batch[0]), *batch[1:]])
        raise TypeError(f"on_inputs: unsupported batch type {type(batch).__name__}")
    return apply


class Compose:
    """Apply callables in order."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomCrop:
    """Class form of random_crop."""

    def __init__(self, size, padding=0):
        self.size, self.padding = size, padding

    def __call__(self, x):
        return random_crop(x, self.size, self.padding)


class RandomHorizontalFlip:
    """Class form of random_horizontal_flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return random_horizontal_flip(x, self.p)


class Normalize:
    """Class form of normalize."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return normalize(x, self.mean, self.std)


class ToFloat:
    """Class form of to_float."""

    def __call__(self, x):
        return to_float(x)


class RepeatChannels:
    """Class form of repeat_channels."""

    def __init__(self, n=3):
        self.n = n

    def __call__(self, x):
        return repeat_channels(x, self.n)


class Resize:
    """Class form of resize."""

    def __init__(self, size, antialias=True):
        self.size, self.antialias = size, antialias

    def __call__(self, x):
        return resize(x, self.size, self.antialias)


class RandomResizedCrop:
    """Class form of random_resized_crop."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
        self.size, self.scale, self.ratio = size, scale, ratio

    def __call__(self, x):
        return random_resized_crop(x, self.size, self.scale, self.ratio)


class RandomAffine:
    """Class form of random_affine."""

    def __init__(self, degrees=10.0, translate=(0.0, 0.0), scale=(1.0, 1.0)):
        self.degrees, self.translate, self.scale = degrees, translate, scale

    def __call__(self, x):
        return random_affine(x, self.degrees, self.translate, self.scale)


class RandomBrightnessContrast:
    """Class form of random_brightness_contrast."""

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness, self.contrast = brightness, contrast

    def __call__(self, x):
        return random_brightness_contrast(x, self.brightness, self.contrast)
