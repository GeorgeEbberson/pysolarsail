"""
Tests for sail_properties.py.
"""
import numpy as np
from parameterized import parameterized

from pysolarsail.sail_properties import SailProperties, ideal_sail, wright_sail
from pysolarsail.units import SOLAR_IRRADIANCE_W_M2, SPEED_OF_LIGHT_M_S

from ..common_test_utils import TestCase

WRIGHT_SAIL = wright_sail(1)
IDEAL_SAIL = ideal_sail(1)


class TestSailProperties(TestCase):
    """Tests for the sail properties routines."""

    @parameterized.expand(
        [
            (1, 0, 0, 0, 0, 1, 1, 1, 0),
            (
                0.88,
                0.94,
                0.79,
                0.55,
                0.05,
                0.55,
                WRIGHT_SAIL.G,
                WRIGHT_SAIL.H,
                WRIGHT_SAIL.K,
            ),
        ]
    )
    def test_properties(
        self,
        rho,
        s,
        B_f,
        B_b,
        eps_f,
        eps_b,
        exp_g,
        exp_h,
        exp_k,
    ):
        """
        Given a set of basic optical coefficients, test that the derived coefficients
        are calculated correctly.
        """
        sail = SailProperties(1, rho, s, B_f, B_b, eps_f, eps_b)
        self.assertAlmostEqual(sail.G, exp_g)
        self.assertAlmostEqual(sail.H, exp_h)
        self.assertAlmostEqual(sail.K, exp_k)

    @parameterized.expand(
        [
            (WRIGHT_SAIL, 0, WRIGHT_SAIL.G + WRIGHT_SAIL.K),
            (WRIGHT_SAIL, np.pi / 2, 0),
            (WRIGHT_SAIL, 1.1, 0.377525508232),
        ]
    )
    def test_Q(self, sail, beta, q):
        """
        For a given sail, check that Q is calculated correctly for different beta.
        Verified against hand calcs.
        """
        self.assertAlmostEqual(sail.Q(beta), q)

    @parameterized.expand(
        [
            (WRIGHT_SAIL, 0, 0),
            (WRIGHT_SAIL, np.pi / 2, 3.07866658201),
            (WRIGHT_SAIL, 1.1, 0.913896143008),
        ]
    )
    def test_delta(self, sail, beta, delta):
        """
        For a given sail check that delta (force cone angle) is calculated correctly.
        """
        self.assertAlmostEqual(sail.delta(beta), delta)

    @parameterized.expand(
        [
            (IDEAL_SAIL, 1, 0, 0, [2, 0, 0]),
            (IDEAL_SAIL, 1, 0, np.pi / 2, [0, 0, 0]),
            # Invented example, checked by hand calculation.
            (WRIGHT_SAIL, 25, 0.1, 1.1, [5.76354111238, 7.43661860982, 0.746150689641]),
            # Example for sail at 1AU.
            (
                IDEAL_SAIL,
                SOLAR_IRRADIANCE_W_M2 / SPEED_OF_LIGHT_M_S,
                0,
                0,
                [2 * SOLAR_IRRADIANCE_W_M2 / SPEED_OF_LIGHT_M_S, 0, 0],
            ),
        ]
    )
    def test_calculate_force(self, sail, srp, alpha, beta, result):
        """Given an SRP and alpha and beta, test that the returned force vector has
        the correct dimensions and values."""
        force = sail.calculate_force(srp, alpha, beta)
        # 1E-17 is mm accurate at AU scale. Needed because of zero errors.
        np.testing.assert_allclose(force, result, atol=1e-17)
        self.assertEqual(force.dtype, np.float64)
        self.assertEqual(force.shape, (3,))

    def test_calculate_force_magnitude_no_alpha_dependence(self):
        """Test that the magnitude of force is the same for all alpha, only the
        direction changes."""

        srp = 1
        beta = 0.1
        mag_0 = np.linalg.norm(WRIGHT_SAIL.calculate_force(srp, 0, beta))

        for alpha in np.arange(0, 2 * np.pi):
            self.assertAlmostEqual(
                mag_0,
                np.linalg.norm(WRIGHT_SAIL.calculate_force(srp, alpha, beta)),
            )
