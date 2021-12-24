"""
Common test things.
"""
import unittest

import numpy as np


# 1-in-a-billion accurate. Small atol to allow for zero comparisons.
# TODO change these to justified values.
FLOAT_EQUAL_PRECISION_RTOL = 1e-9
FLOAT_EQUAL_PRECISION_ATOL = 1e-16


class TestCase(unittest.TestCase):
    """Testcase overrides and extra convenience methods."""

    @staticmethod
    def assertArrayEqual(first: np.ndarray, second: np.ndarray) -> None:
        """Wrapper for the numpy equivalent to make the names consistent."""
        np.testing.assert_allclose(
            first,
            second,
            rtol=FLOAT_EQUAL_PRECISION_RTOL,
            atol=FLOAT_EQUAL_PRECISION_ATOL,
            equal_nan=False,
        )
