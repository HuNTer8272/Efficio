"""Efficio: get more training out of every GPU hour on Colab, Kaggle and local GPUs.

    import efficio
    run = efficio.Run(budget="2h")
    model, optimizer = run.prepare(model, optimizer)
    for x, y in run.loop(loader, epochs=20):
        with run.autocast():
            loss = loss_fn(model(x), y)
        run.backward(loss)
        run.step()
    run.report()
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .data import CachedImageDataset, ImageCache
from .env import detect_hardware, detect_platform
from .interfaces import Budget, HardwareInfo, TrainState
from .run import Run
from .speed import gpu_augment
from .speed.gpu_augment import on_inputs
from .speed.gpu_data import GPULoader

try:
    __version__ = _version("efficio")       # the version in pyproject.toml, once installed
except PackageNotFoundError:                 # running from a source tree that isn't installed
    __version__ = "0+unknown"

__all__ = [
    "Run",
    "GPULoader",
    "ImageCache",
    "CachedImageDataset",
    "on_inputs",
    "gpu_augment",
    "detect_hardware",
    "detect_platform",
    "Budget",
    "HardwareInfo",
    "TrainState",
    "__version__",
]
