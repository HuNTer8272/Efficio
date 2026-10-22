"""Efficio: get more training out of every GPU hour."""

from .data import CachedImageDataset, ImageCache
from .env import detect_hardware, detect_platform
from .interfaces import Budget, HardwareInfo, TrainState
from .run import Run
from .speed.gpu_augment import on_inputs
from .speed.gpu_data import GPULoader

__version__ = "0.0.1"
