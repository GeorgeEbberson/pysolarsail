"""
Utilities for improving/working with numba.
"""
import os


def class_spec(cls):
    """Check that the first element of each arg is a string, and return a list."""
    if os.environ["NUMBA_DISABLE_JIT"] == "1":
        return cls
    else:
        return cls.class_type.instance_type