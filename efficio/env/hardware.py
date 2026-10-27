"""Detect the GPU and what it supports."""

from ..interfaces import HardwareInfo
from .detect import SESSION_LIMITS, detect_platform


def detect_hardware() -> HardwareInfo:
    """Platform, GPU name, count, memory and compute capability, as a HardwareInfo."""
    platform = detect_platform()
    info = HardwareInfo(platform=platform, device="cpu",
                        session_limit_hours=SESSION_LIMITS[platform])
    try:
        import torch
    except ImportError:
        return info
    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    info.device = "cuda"
    info.gpu_name = props.name
    info.gpu_count = torch.cuda.device_count()
    info.vram_gb = round(props.total_memory / 1024**3, 1)
    info.compute_capability = cc
    # Newer PyTorch reports bf16 as "supported" on a T4 through slow emulation,
    # so check the hardware generation instead: native bf16 starts at Ampere (8.0).
    info.bf16_native = cc >= (8, 0)
    return info


def describe(info: HardwareInfo) -> str:
    """One log line describing the hardware and the chosen precision."""
    if info.device == "cpu":
        return f"[efficio] {info.platform}: no GPU found, running on CPU (fp32)"
    major, minor = info.compute_capability
    return (f"[efficio] {info.platform}: {info.gpu_count}x {info.gpu_name} "
            f"({info.vram_gb} GB, compute {major}.{minor}) "
            f"-> using {info.precision}")
