"""
Tests for spacecraft.py.
"""
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from parameterized import parameterized

from pysolarsail.spacecraft import (
    solarsail_acceleration,
    unit_vector_and_mag,
)

from ..common_test_utils import TestCase


ONE_OVER_ROOT_3 = [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]


def add_prop_mock(obj, name, **kwargs):
    """Add a property mock and store it in obj.mocks."""
    if not hasattr(obj, "mocks"):
        obj.mocks = {}
    prop = PropertyMock(**kwargs)
    setattr(type(obj), name, prop)
    obj.mocks[name] = prop


def mock_spacecraft(aqf, pos, vel, mass=1):
    """Mock a spacecraft object. Positional args should be iterables and will return
    the next element on each usage."""

    spacecraft = MagicMock()
    spacecraft.sail = MagicMock()
    spacecraft.sail.aqf = MagicMock(side_effect=aqf)

    add_prop_mock(spacecraft, "pos_m", side_effect=pos)
    add_prop_mock(spacecraft, "vel_m_s", side_effect=vel)
    add_prop_mock(spacecraft, "mass_kg", return_value=mass)

    return spacecraft


def mock_body(pos, gravity=True, radiation=0, grav_param=1):
    """Return a mock for a spicebody. Positional args should be iterables and the
    next element will be returned on each usage."""

    body = MagicMock()

    add_prop_mock(body, "gravity", return_value=gravity)
    add_prop_mock(body, "radiation_w_m2", return_value=radiation)
    add_prop_mock(body, "is_star", return_value=True if radiation != 0 else False)
    add_prop_mock(body, "gravitation_parameter_m3_s2", return_value=grav_param)
    add_prop_mock(body, "pos_m", side_effect=pos)

    return body


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

    def test_solarsail_acceleration_no_bodies(self):
        """Check that accel is zero for no bodies."""
        sc = mock_spacecraft([None], [None], [None])
        accel = solarsail_acceleration(sc, [])
        self.assertArrayEqual(accel, np.array([0, 0, 0]))

        # Use this as the metric for not calculating anything (i.e. not zero by chance).
        sc.sail.aqf.assert_not_called()

    def test_solarsail_acceleration_no_gravity_short_circuit(self):
        """Check that gravity turned off gives zero accel."""
        sc = mock_spacecraft([None], [None], [None])
        bd = mock_body([None], gravity=False)
        accel = solarsail_acceleration(sc, [bd])
        self.assertArrayEqual(accel, np.array([0, 0, 0]))

        # Check that the zero is not by chance.
        bd.mocks["radiation_w_m2"].assert_not_called()

    def test_solarsail_acceleration(self):
        """Check that solarsail acceleration returns sensible values."""
