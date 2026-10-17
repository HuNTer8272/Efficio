"""Detect the GPU and what it supports."""

from ..interfaces import HardwareInfo
from .detect import detect_platform, SESSION_LIMITS


def detect_hardware() -> HardwareInfo:
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
    if info.device == "cpu":
        return f"[efficio] {info.platform}: no GPU found, running on CPU (fp32)"
    return (f"[efficio] {info.platform}: {info.gpu_count}x {info.gpu_name} "
            f"({info.vram_gb} GB, compute {info.compute_capability[0]}.{info.compute_capability[1]}) "
            f"-> using {info.precision}")
