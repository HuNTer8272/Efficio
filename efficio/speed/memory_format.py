"""channels_last memory format for convolutional models on the GPU.

Tensor cores on T4 and newer GPUs run convolutions faster when images are stored
as NHWC (channels_last) instead of NCHW.
"""

from torch import nn


def has_conv2d(model):
    """True if the model contains any nn.Conv2d layer."""
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


def use_channels_last(setting, device, model):
    """Returns (use, reason). `setting` is "auto", True or False."""
    if setting not in ("auto", True, False):
        raise ValueError(f'channels_last must be "auto", True or False, not {setting!r}')
    if setting is False:
        return False, "turned off"
    if device != "cuda":
        return False, "no GPU"
    if not has_conv2d(model):
        return False, "model has no Conv2d layers"
    return True, "conv model on GPU"
