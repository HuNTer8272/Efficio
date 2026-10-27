"""Planned for a future release: releasing the runtime when training ends (for example on Colab).

Nothing in this module is usable yet; any name looked up here raises NotImplementedError.
"""

from .._planned import planned

__getattr__ = planned(__name__)
