"""
Tests for spacecraft.py.
"""
from collections import namedtuple
from itertools import product
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from parameterized import parameterized

from pysolarsail.spacecraft import (
    compute_k,
    rkf_step,
    solarsail_acceleration,
    solve_rkf,
    unit_vector_and_mag,
)
from pysolarsail.units import M_PER_AU, SPEED_OF_LIGHT_M_S

from ..common_test_utils import TestCase, skip

# Test cases for the RKF method.
RkfTestCase = namedtuple(
    "RkfTestCase",
    ["x0", "y0", "dx", "y1", "k", "ykplus1", "zkplus1", "fn"],
)


def d_dx_x2_sinx(x, *_):
    """The derivative of x^2 * sin(x) wrt x."""
    return (2 * x * np.sin(x)) + (x * x * np.cos(x))


# https://www.desmos.com/calculator/hzijoloamv
RKF_TEST_CASE_1D = RkfTestCase(
    x0=1,
    y0=0.841470984808,
    dx=1,
    y1=3.6371897073,
    k=[
        2.22324427548,
        2.86515273963,
        3.06527266723,
        2.33392213663,
        1.97260236111,
        3.15164366356,
    ],
    ykplus1=3.63630583173,
    zkplus1=3.63660957909,
    fn=d_dx_x2_sinx,
)

RKF_TEST_CASES = [
    ("x^2 * sin(x)", RKF_TEST_CASE_1D),
]

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
        sc = mock_spacecraft(aqf=[None], pos=[None], vel=[None])
        accel = solarsail_acceleration(sc, [], np.zeros((2, 3)))
        self.assertArrayEqual(accel, np.array([0, 0, 0]))

        # Use this as the metric for not calculating anything (i.e. not zero by chance).
        sc.sail.aqf.assert_not_called()

    def test_solarsail_acceleration_no_gravity_short_circuit(self):
        """Check that gravity turned off gives zero accel."""
        sc = mock_spacecraft(aqf=[None], pos=[None], vel=[None])
        bd = mock_body([None], gravity=False)
        accel = solarsail_acceleration(sc, [bd], np.zeros((2, 3)))
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

        accel = solarsail_acceleration(sc, [bd], np.array([sc_pos, [0, 0, 0]]))
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

        sc = mock_spacecraft(aqf=[aqf], pos=[sc_pos], vel=[None], mass=mass)

        # Gravity must be True because otherwise the body is skipped. (Stars are
        # expected to always have gravity on).
        bd = mock_body([[0, 0, 0]], gravity=True, grav_param=0, radiation=radiation)

        accel = solarsail_acceleration(sc, [bd], np.array([sc_pos, [0, 0, 0]]))
        self.assertArrayEqual(accel, np.array(exp_accel))

    def test_solarsail_acceleration_gravity_multiple_bodies(self):
        """Test that gravity of several bodies is summed correctly."""
        coords = list(product([-1, 1], [-1, 1], [-1, 1]))
        sc = mock_spacecraft(
            aqf=[None],
            pos=[[0, 0, 0] for _ in range(len(coords))],
            vel=[None],
        )

        bds = list(mock_body([list(x)]) for x in coords)
        accel = solarsail_acceleration(sc, bds, np.zeros((2, 3)))
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
        bodies = [
            # pos, aqf, radiation
            ([1, 0, 0], [-1, 0, 0], 1),
            ([-ROOT_2_ON_2, ROOT_2_ON_2, 0], [1, 0, 0], 0.5),
            ([-ROOT_2_ON_2, -ROOT_2_ON_2, 0], [1, 0, 0], 0.5),
        ]

        sc = mock_spacecraft(
            [body[1] for body in bodies],
            [[0, 0, 0] for _ in range(len(bodies))],
            [None],
            alpha=[0 for _ in bodies],
            beta=[0 for _ in bodies],
        )

        bds = [
            mock_body([pos], radiation=rad, grav_param=0) for pos, _, rad in bodies
        ]
        accel = solarsail_acceleration(sc, bds, np.zeros((2, 3)))
        exp_accel = [0, 0, 0]
        self.assertArrayAlmostEqual(
            accel,
            np.array(exp_accel, dtype=np.float64),
            atol=0.01,
        )

    @parameterized.expand(RKF_TEST_CASES)
    def test_compute_k(self, _, case):
        """All values of X should be correct for a given k value.

        Test function is x^2 * sin(x) so d/dx = 2x*sin(x) + cos(x)*x^2.
        """

        with patch("pysolarsail.spacecraft.rkf_rhs", wraps=case.fn) as fn:
            k = compute_k(case.x0, case.y0, case.dx, None, None)

        for idx, kx in enumerate(case.k):
            self.assertArrayAlmostEqual(
                kx,
                k[idx, :, :],
                err_msg=f"Mismatch on k[{idx}, :, :]",
            )

    @parameterized.expand(RKF_TEST_CASES)
    def test_rkf_step(self, _, case):
        """y and z should be correct."""

        with patch("pysolarsail.spacecraft.rkf_rhs", wraps=case.fn) as fn:
            y_kplus1, z_kplus1 = rkf_step(case.dx, case.x0, case.y0, None, None)

        self.assertArrayAlmostEqual(y_kplus1, case.ykplus1)
        self.assertArrayAlmostEqual(z_kplus1, case.zkplus1)

    @parameterized.expand(RKF_TEST_CASES)
    @skip("#3", "RKF testing approach is undecided")
    def test_solve_rkf(self, _, case):
        """RKF should be correct."""

        with patch(
            "pysolarsail.spacecraft.get_init_state"
        ) as init, patch(
            "pysolarsail.spacecraft.rkf_rhs", wraps=case.fn
        ) as rhs_fn:
            init.return_value = np.ones((2, 3)) * case.x0
            results = solve_rkf(
                None,
                case.x0,
                2.1,
                [mock_body([[0, 0, 0], [0, 0, 0], [0, 0, 0]])],
                init_time_step=case.dx,
            )

        print(results)
        print(results[:, 16] - results[:, 10])
        self.assertEqual(results, 0)