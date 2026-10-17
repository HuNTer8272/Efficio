"""Shared data objects. Both teammates build against these, so change them together."""

from dataclasses import dataclass, field, asdict
import time


@dataclass
class HardwareInfo:
    platform: str                 # "colab", "kaggle" or "local"
    device: str                   # "cuda" or "cpu"
    gpu_name: str = "none"        # e.g. "Tesla T4"
    gpu_count: int = 0
    vram_gb: float = 0.0
    compute_capability: tuple = (0, 0)
    bf16_native: bool = False     # True only on Ampere (8.0) or newer
    session_limit_hours: float = 0.0   # 0 = no known limit (local)

    @property
    def precision(self) -> str:
        """Best training precision for this hardware."""
        if self.device != "cuda":
            return "fp32"
        if self.bf16_native:
            return "bf16"
        return "fp16"             # T4, P100 and older: fp16 + GradScaler


@dataclass
class Budget:
    total_seconds: float = 0.0             # 0 = no budget
    safety_margin_seconds: float = 600.0   # stop and save this long before the end
    started_at: float = field(default_factory=time.time)

    @classmethod
    def parse(cls, text, safety_margin_seconds=600.0):
        """Accepts "2h", "90m", "1h30m", "3600s" or a number of seconds."""
        if text is None:
            return cls(0.0, safety_margin_seconds)
        if isinstance(text, (int, float)):
            return cls(float(text), safety_margin_seconds)
        total, num = 0.0, ""
        units = {"h": 3600, "m": 60, "s": 1}
        for ch in text.strip().lower():
            if ch.isdigit() or ch == ".":
                num += ch
            elif ch in units and num:
                total += float(num) * units[ch]
                num = ""
            elif ch != " ":
                raise ValueError(f"Can't read budget '{text}'. Use e.g. '2h' or '90m'.")
        if num:                      # bare number = seconds
            total += float(num)
        return cls(total, safety_margin_seconds)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def remaining(self) -> float:
        if self.total_seconds <= 0:
            return float("inf")
        return self.total_seconds - self.elapsed()

    def nearly_out(self) -> bool:
        return self.remaining() <= self.safety_margin_seconds


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0                  # optimizer steps taken in total
    batch_in_epoch: int = 0        # batches already done in the current epoch
    batch_size: int = 0
    grad_accum_steps: int = 1

    def to_dict(self):
        return asdict(self)
