"""Detect the platform (Colab, Kaggle or local) and the GPU."""

from .detect import default_checkpoint_dir, detect_platform
from .hardware import describe, detect_hardware

__all__ = ["detect_platform", "default_checkpoint_dir", "detect_hardware", "describe"]
