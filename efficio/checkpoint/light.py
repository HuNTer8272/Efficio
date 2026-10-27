"""Planned for a future release: light checkpoints (trainable parameters only, fp16 weights,
8-bit optimizer state).

Nothing in this module is usable yet; any name looked up here raises NotImplementedError.
"""

from .._planned import planned

__getattr__ = planned(__name__)
