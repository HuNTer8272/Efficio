"""Efficio: get more training out of every GPU hour."""

from .env import detect_hardware, detect_platform
from .interfaces import Budget, HardwareInfo, TrainState
from .run import Run

__version__ = "0.0.1"
