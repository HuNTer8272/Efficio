"""Planned for a future release: a budget planner that times the first steps and fits the number
of epochs and the learning-rate schedule to the time budget.

Nothing in this module is usable yet; any name looked up here raises NotImplementedError.
"""

from .._planned import planned

__getattr__ = planned(__name__)
