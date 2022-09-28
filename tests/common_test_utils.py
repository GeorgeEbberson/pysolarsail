"""
Common test things.
"""
import logging
import os
import unittest

import numpy as np
import pytest
from parameterized import parameterized

LOGGER = logging.getLogger(__name__)

# 1-in-a-billion accurate. Small atol to allow for zero comparisons.
# TODO change these to justified values.
FLOAT_EQUAL_PRECISION_RTOL = 1e-9
FLOAT_EQUAL_PRECISION_ATOL = 1e-16

# Strings to be considered true and false respectively. Should be all lowercase.
TRUE_STRINGS = ("true", "y", "1")
FALSE_STRINGS = ("false", "n", "0")


def load_env_variable_true_false(name):
    """Loads an environment variable and tries to cast it to true/false."""
    env_var = os.getenv(name, "false")
    if env_var.lower() in TRUE_STRINGS:
        res = True
    elif env_var.lower() in FALSE_STRINGS:
        res = False
    else:
        res = False
    LOGGER.info(f"{name} is {res}")
    return res


def skip(gh_issue, description):
    """Mark a test as skipped against the relevant GitHub issue."""
    return pytest.mark.skip(reason=f"{gh_issue}: {description}")


def cases(iter_of_cases):
    """Expand cases strictly on a test class (i.e. cases are methods)."""
    return parameterized.expand(
        [x if isinstance(x, tuple) else (x,) for x in iter_of_cases]
    )


class TestCase(unittest.TestCase):
    """Testcase overrides and extra convenience methods."""

    SHOULD_PLOT = load_env_variable_true_false("PYSOLARSAIL_DO_PLOT")

    @staticmethod
    def assertArrayEqual(
        first: np.ndarray,
        second: np.ndarray,
        rtol=FLOAT_EQUAL_PRECISION_RTOL,
        atol=FLOAT_EQUAL_PRECISION_ATOL,
        equal_nan=False,
    ) -> None:
        """Wrapper for the numpy equivalent to make the names consistent."""
        np.testing.assert_allclose(
            first,
            second,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    @staticmethod
    def assertArrayAlmostEqual(
        first: np.ndarray,
        second: np.ndarray,
        rtol=FLOAT_EQUAL_PRECISION_RTOL,
        atol=FLOAT_EQUAL_PRECISION_ATOL,
        equal_nan=False,
        **kwargs,
    ) -> None:
        """Wrapper for allclose with relaxed tolerances."""
        np.testing.assert_allclose(
            first,
            second,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            **kwargs,
        )
