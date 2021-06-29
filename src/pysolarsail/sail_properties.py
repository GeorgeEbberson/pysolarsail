"""
Stores and handles sail optical properties and generates the sail properties object.
"""
from numba import float64
from numba.experimental import jitclass


@jitclass
class SailProperties(object):
    """Calculate the sail properties and get the non-ideal force vector."""

    G: float64
    H: float64
    K: float64

    def __init__(
        self,
        area: float,
        rho: float,
        s: float,
        B_b: float,
        B_f: float,
        eps_f: float,
        eps_b: float,
    ):
        """Initialise an instance."""
        self.G = 1 + s * rho
        self.H = 1 - s * rho
        self.K = B_f * (1 - s) * rho + (1 - rho) * (
            (eps_f * B_f - eps_b * B_b) / (eps_f + eps_b)
        )
