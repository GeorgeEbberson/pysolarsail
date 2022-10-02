"""
Utilities for improving/working with numba.
"""
import os

try:
    _JITTING = False if os.environ["NUMBA_DISABLE_JIT"] == "1" else True
except KeyError:
    _JITTING = True


def class_spec(cls):
    """Check that the first element of each arg is a string, and return a list."""
    if not _JITTING:
        return cls
    else:
        return cls.class_type.instance_type
