"""
Tests for spacecraft.py.
"""
import numpy as np
import pytest
from parameterized import parameterized

from pysolarsail.spacecraft import (
    unit_vector_and_mag,
)

from ..common_test_utils import TestCase


ONE_OVER_ROOT_3 = [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]


@pytest.mark.unit
class TestSolarSailcraft(TestCase):
    """Tests for the SolarSailcraft class."""


@pytest.mark.unit
class TestRkf(TestCase):
    """
    Tests for the Runge-Kutta-Fehlberg methods.
    """

    @parameterized.expand(
        [
            ([1, 0, 0], [1, 0, 0], 1),
            (ONE_OVER_ROOT_3, ONE_OVER_ROOT_3, 1),
            ([1, 1, 1], ONE_OVER_ROOT_3, np.sqrt(3)),
        ]
    )
    def test_unit_vector_and_mag(self, in_vec, exp_vec, exp_mag):
        """Check that the proper unit vector and magnitude are returned."""
        unit, mag = unit_vector_and_mag(np.array(in_vec))
        np.testing.assert_allclose(unit, exp_vec)
        self.assertEqual(exp_mag, mag)
