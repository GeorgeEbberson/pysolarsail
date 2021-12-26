"""
Tests for spacecraft.py.
"""
from itertools import product
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from parameterized import parameterized

from pysolarsail.spacecraft import (
    get_init_state,
    rkf_rhs,
    solarsail_acceleration,
    unit_vector_and_mag,
)
from pysolarsail.units import M_PER_AU, SPEED_OF_LIGHT_M_S

from ..common_test_utils import TestCase


ONE_OVER_ROOT_3 = [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]
ROOT_2_ON_2 = np.sqrt(2) / 2

SRP_0 = (M_PER_AU ** 2) / SPEED_OF_LIGHT_M_S


def add_prop_mock(obj, name, **kwargs):
    """Add a property mock and store it in obj.mocks."""
    if not hasattr(obj, "mocks"):
        obj.mocks = {}
    prop = PropertyMock(**kwargs)
    setattr(type(obj), name, prop)
    obj.mocks[name] = prop


def mock_spacecraft(aqf, pos, vel, mass=1, alpha=None, beta=None):
    """Mock a spacecraft object. Positional args should be iterables and will return
    the next element on each usage."""

    spacecraft = MagicMock()
    spacecraft.sail = MagicMock()
    spacecraft.sail.aqf = MagicMock(side_effect=[np.array(x, dtype=np.float64) for x in aqf])

    add_prop_mock(spacecraft, "pos_m", side_effect=[np.array(x, dtype=np.float64) for x in pos])
    add_prop_mock(spacecraft, "vel_m_s", side_effect=[np.array(x, dtype=np.float64) for x in vel])
    add_prop_mock(spacecraft, "mass", return_value=mass)
    add_prop_mock(spacecraft, "alpha_rad", side_effect=[0] if alpha is None else alpha)
    add_prop_mock(spacecraft, "beta_rad", side_effect=[0] if beta is None else beta)

    return spacecraft


def mock_body(pos: list, gravity=True, radiation=0, grav_param=1):
    """Return a mock for a spicebody. Positional args should be iterables and the
    next element will be returned on each usage."""

    body = MagicMock()

    add_prop_mock(body, "gravity", return_value=gravity)
    add_prop_mock(body, "radiation_w_m2", return_value=radiation)
    add_prop_mock(body, "is_star", return_value=True if radiation != 0 else False)
    add_prop_mock(body, "gravitation_parameter_m3_s2", return_value=grav_param)
    add_prop_mock(body, "pos_m", side_effect=[np.array(x, dtype=np.float64) for x in pos])

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
        self.assertArrayEqual(unit, exp_vec)
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

    @parameterized.expand(
        [
            ([1, 0, 0], 1, [-1, 0, 0]),
            ([1, 1, 1], 1, [-x / 3 for x in ONE_OVER_ROOT_3]),
            ([2, 3, 6], 2, [-x * (2 / 7) / 49 for x in [2, 3, 6]]),
        ]
    )
    def test_solarsail_acceleration_gravity_single_body(
        self,
        sc_pos,
        grav_param,
        exp_accel,
    ):
        """Check that gravity is calculated properly."""

        sc = mock_spacecraft([None], [sc_pos], [None], mass=1)
        bd = mock_body([[0, 0, 0]], gravity=True, grav_param=grav_param)

        accel = solarsail_acceleration(sc, [bd])
        self.assertArrayEqual(accel, np.array(exp_accel))

        # Check that sail accel code was never called.
        sc.sail.aqf.assert_not_called()

    @parameterized.expand(
        [
            ([1, 0, 0], [1, 0, 0], 1, 1, [SRP_0, 0, 0]),
        ]
    )
    def test_solarsail_acceleration_srp_single_body(
        self,
        aqf,
        sc_pos,
        mass,
        radiation,
        exp_accel,
    ):
        """Check that force on the sail itself is calculated properly."""

        sc = mock_spacecraft([aqf], [sc_pos], [None], mass=mass)

        # Gravity must be True because otherwise the body is skipped. (Stars are
        # expected to always have gravity on).
        bd = mock_body([[0, 0, 0]], gravity=True, grav_param=0, radiation=radiation)

        accel = solarsail_acceleration(sc, [bd])
        self.assertArrayEqual(accel, np.array(exp_accel))

    def test_solarsail_acceleration_gravity_multiple_bodies(self):
        """Test that gravity of several bodies is summed correctly."""
        coords = list(product([-1, 1], [-1, 1], [-1, 1]))
        sc = mock_spacecraft(
            [None],
            [[0, 0, 0] for _ in range(len(coords))],
            [None],
        )

        bds = list(mock_body([list(x)]) for x in coords)
        accel = solarsail_acceleration(sc, bds)
        exp_accel = [0, 0, 0]
        self.assertArrayEqual(accel, np.array(exp_accel, dtype=np.float64))

    def test_solarsail_acceleration_srp_multiple_bodies(self):
        """Test that SRP of several bodies is summed correctly.

        This test is set up with one body directly in front of the craft, and two bodies
        behind it at 45 degrees. Each of the two behind is half as radiative as the one
        in front, meaning that they exert half as much force (because we mock aqf to
        give a perfect sail).
        """
        # These must be unit vectors, or below code changed.
        coords = [
            ([1, 0, 0], [-1, 0, 0], 1),
            ([-ROOT_2_ON_2, ROOT_2_ON_2, 0], [1, 0, 0], 0.5),
            ([-ROOT_2_ON_2, -ROOT_2_ON_2, 0], [1, 0, 0], 0.5),
        ]

        sc = mock_spacecraft(
            [x for _, x, _ in coords],
            [[0, 0, 0] for _ in coords],
            [None],
            alpha=[0 for _ in coords],
            beta=[0 for _ in coords],
        )

        bds = [
            mock_body([pos], radiation=rad, grav_param=0) for pos, _, rad in coords
        ]
        accel = solarsail_acceleration(sc, bds)
        exp_accel = [0, 0, 0]
        self.assertArrayEqual(accel, np.array(exp_accel, dtype=np.float64))

    def test_rkf_rhs(self):
        """Check that rkf_rhs sets time then stacks the output correctly.

        We care that the time is set for everything (i.e. we're working on the most
        up-to-date state, and that the output has the input in the correct places.
        """
        craft = mock_spacecraft([-1, 0, 0], [0, 0, 0], [None], [0], [0])
        bds = [mock_body([1, 0, 0])]
        state = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        time = float(123456)
        with patch(
            "pysolarsail.spacecraft.solarsail_acceleration",
            return_value=np.array([7, 8, 9]),
        ):
            retval = rkf_rhs(time, state, bds, craft)
        exp = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float64)
        self.assertArrayEqual(exp, retval)
        self.assertTrue(craft.set_time.called)
        self.assertTrue(all([x.set_time.called] for x in bds))

    def test_get_init_state(self):
        """Simple check that the starting state is correct."""
        craft = mock_spacecraft([None], [[1, 2, 3]], [[4, 5, 6]])
        retval = get_init_state(craft)
        self.assertArrayEqual(retval, np.array([[1, 2, 3], [4, 5, 6]]))