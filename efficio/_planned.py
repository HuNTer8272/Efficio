"""Support for modules whose features are planned but not released yet.

A planned module sets `__getattr__ = planned(__name__)`. Looking up any public name in it
raises NotImplementedError, so nothing in it can be mistaken for a working feature.
"""

MESSAGE = "planned for a future release"


def planned(module_name):
    """Return a module-level __getattr__ that rejects every public name."""
    def __getattr__(name):
        if name.startswith("__"):          # let Python and tools probe dunder names safely
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        raise NotImplementedError(f"{module_name}.{name}: {MESSAGE}")
    return __getattr__
