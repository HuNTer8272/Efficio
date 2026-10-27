"""Work out where we are running: Colab, Kaggle or a local machine."""

import importlib.util
import os

# Known session caps in hours (approximate; Colab free limits vary and are not guaranteed).
SESSION_LIMITS = {"colab": 12.0, "kaggle": 12.0, "local": 0.0}


def detect_platform() -> str:
    """Where we are running: "colab", "kaggle" or "local"."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or _has_module("google.colab"):
        return "colab"
    return "local"


def _has_module(name):
    # find_spec("google.colab") raises instead of returning None when "google" itself is missing
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def default_checkpoint_dir(platform: str) -> str:
    """A sensible place to save checkpoints on each platform."""
    if platform == "kaggle":
        return "/kaggle/working/efficio_checkpoints"   # kept in the notebook's output
    if platform == "colab":
        if os.path.isdir("/content/drive/MyDrive"):     # Drive mounted: survives the session
            return "/content/drive/MyDrive/efficio_checkpoints"
        return "/content/efficio_checkpoints"
    return os.path.join(os.getcwd(), "efficio_checkpoints")
