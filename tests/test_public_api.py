"""The public API of the release, and planned modules that must not look usable."""

import importlib

import pytest

import efficio

PLANNED = [
    "efficio.adapters.hf_trainer",
    "efficio.adapters.lightning",
    "efficio.budget.planner",
    "efficio.preflight",
    "efficio.report.notify",
    "efficio.speed.amp",
    "efficio.speed.batch_size",
    "efficio.speed.cache",
    "efficio.speed.multigpu",
]


def test_all_names_exist():
    expected = {"Run", "GPULoader", "ImageCache", "CachedImageDataset", "on_inputs",
                "gpu_augment", "detect_hardware", "detect_platform", "Budget", "HardwareInfo",
                "TrainState", "__version__"}
    assert set(efficio.__all__) == expected
    for name in efficio.__all__:
        assert hasattr(efficio, name), name
    assert isinstance(efficio.__version__, str) and efficio.__version__


def test_gpu_augment_is_the_module():
    from efficio.speed import gpu_augment
    assert efficio.gpu_augment is gpu_augment
    assert callable(efficio.gpu_augment.RandomCrop(32, padding=4))


@pytest.mark.parametrize("module_name", PLANNED)
def test_planned_modules_raise(module_name):
    module = importlib.import_module(module_name)
    with pytest.raises(NotImplementedError, match="planned for a future release"):
        module.anything()
    assert not hasattr(module, "__wrapped__")      # dunder probes stay ordinary


def test_nothing_planned_is_exported():
    planned_words = ("planner", "shutdown", "notify", "preflight", "storage", "light",
                     "multigpu", "batch_size", "adapters")
    assert not any(w in name.lower() for name in efficio.__all__ for w in planned_words)
