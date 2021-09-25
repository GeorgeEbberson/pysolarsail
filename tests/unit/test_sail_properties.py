"""
Tests for sail_properties.py.
"""
from collections import namedtuple

import numpy as np
import pytest
from parameterized import parameterized

from pysolarsail.sail_properties import SailProperties, ideal_sail, wright_sail

from ..common_test_utils import TestCase

SailPropertiesCase = namedtuple(
    "SailPropertiesCase",
    ["alpha", "beta", "Q", "delta", "aqf"],
)

WRIGHT_SAIL = wright_sail(1)
IDEAL_SAIL = ideal_sail(1)
WRIGHT_CASE1 = SailPropertiesCase(
    alpha=0.1,
    beta=1.1,
    Q=0.377525508232,
    delta=0.913896143008,
    aqf=[0.230541644495, 0.297464744393, 0.0298460275856],
)


@pytest.mark.unit
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
            (WRIGHT_SAIL, WRIGHT_CASE1.beta, WRIGHT_CASE1.Q),
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
            (WRIGHT_SAIL, WRIGHT_CASE1.beta, WRIGHT_CASE1.delta),
        ]
    )
    def test_delta(self, sail, beta, delta):
        """
        For a given sail check that delta (force cone angle) is calculated correctly.
        """
        self.assertAlmostEqual(sail.delta(beta), delta)

    @parameterized.expand(
        [
            (IDEAL_SAIL, 0, 0, [2, 0, 0]),
            (IDEAL_SAIL, 0, np.pi / 2, [0, 0, 0]),
            # Invented example, checked by hand calculation.
            (WRIGHT_SAIL, WRIGHT_CASE1.alpha, WRIGHT_CASE1.beta, WRIGHT_CASE1.aqf),
        ]
    )
    def test_aqf(self, sail, alpha, beta, result):
        """Given an SRP and alpha and beta, test that the returned force vector has
        the correct dimensions and values."""
        force = sail.aqf(alpha, beta)
        self.assertArrayEqual(force, result)
        self.assertEqual(force.dtype, np.float64)
        self.assertEqual(force.shape, (3,))

    def test_aqf_magnitude_no_alpha_dependence(self):
        """Test that the magnitude of force is the same for all alpha, only the
        direction changes."""

        mag_0 = np.linalg.norm(WRIGHT_SAIL.aqf(0, WRIGHT_CASE1.beta))

        for alpha in np.arange(0, 2 * np.pi, step=0.1):
            self.assertAlmostEqual(
                mag_0,
                np.linalg.norm(WRIGHT_SAIL.aqf(alpha, WRIGHT_CASE1.beta)),
            )
