"""Planned for a future release: a CPU pre-flight check that runs a few batches and a checkpoint
round trip before the GPU run.

Nothing in this module is usable yet; any name looked up here raises NotImplementedError.
"""

from ._planned import planned

__getattr__ = planned(__name__)
