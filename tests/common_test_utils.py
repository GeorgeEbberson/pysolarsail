"""
Common test things.
"""
import unittest

import numpy as np


class TestCase(unittest.TestCase):
    """Testcase overrides and extra convenience methods."""

    @staticmethod
    def assertArrayEqual(first: np.ndarray, second: np.ndarray) -> None:
        """Wrapper for the numpy equivalent to make the names consistent."""
        np.testing.assert_array_equal(first, second)
