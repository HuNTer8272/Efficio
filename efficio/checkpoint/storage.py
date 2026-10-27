"""Planned for a future release: storing checkpoints on the Hugging Face Hub or Google Drive, so
a run can move between platforms.

Nothing in this module is usable yet; any name looked up here raises NotImplementedError.
"""

from .._planned import planned

__getattr__ = planned(__name__)
