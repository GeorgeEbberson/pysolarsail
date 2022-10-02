"""
Stores and handles sail optical properties and generates the sail properties object.
"""
import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from pysolarsail.units import SOLAR_RADIATION_PRESSURE_P0

SAIL_PROPERTIES_SPEC = [
    ("area_m2", float64),
    ("G", float64),
    ("H", float64),
    ("K", float64),
]


@jitclass(SAIL_PROPERTIES_SPEC)
class SailProperties(object):
    """Calculate the sail properties and get the non-ideal force vector.

    G, H, and K are properties used to simplify intermediate expressions. Their full
    definitions are stated by Dachwald (2004) amongst other authors.
    """

    def __init__(
        self,
        area_m2: float,
        reflection_coeff_front: float,
        specular_reflection_factor: float,
        nonlambertian_coeff_front: float,
        nonlambertian_coeff_back: float,
        emission_coeff_front: float,
        emission_coeff_back: float,
    ):
        """Initialise an instance."""
        self.area_m2 = area_m2
        self.G = 1 + specular_reflection_factor * reflection_coeff_front
        self.H = 1 - specular_reflection_factor * reflection_coeff_front
        self.K = (
            nonlambertian_coeff_front
            * (1 - specular_reflection_factor)
            * reflection_coeff_front
        ) + (
            (1 - reflection_coeff_front)
            * (
                emission_coeff_front * nonlambertian_coeff_front
                - emission_coeff_back * nonlambertian_coeff_back
            )
            / (emission_coeff_front + emission_coeff_back)
        )

    def Q(self, beta: float) -> float:
        """Calculate the non-ideal force terms based on the optical properties."""
        return np.sqrt(  # type: ignore
            (self.G * np.cos(beta) + self.K) ** 2 + (self.H * np.sin(beta)) ** 2
        ) * np.cos(beta)

    def delta(self, beta: float) -> float:
        """Calculate the force cone angle (not the same as the sail cone angle).

        TODO Further investigation of extreme values as shown in:
        https://www.desmos.com/calculator/1rcf9l10ly
        """
        return beta - np.arctan(  # type: ignore
            (self.H * np.sin(beta)) / (self.G * np.cos(beta) + self.K)
        )

    def aqf(self, alpha: float, beta: float) -> np.ndarray:
        """Calculate area * Q(beta) * f as a convenience."""
        delta = self.delta(beta)

        # This assumes that alpha = gamma, or that the spacecraft clock angle is
        # equal to the force clock angle. This is probably untrue but the assumption
        # is used sometimes in literature so it's acceptable for now.
        # TODO Test whether this assumption is true by deriving it.
        return (
            self.area_m2
            * self.Q(beta)
            * np.array(
                [
                    np.cos(delta),
                    np.sin(delta) * np.cos(alpha),
                    np.sin(delta) * np.sin(alpha),
                ]
            )
        )

    def mass_for_char_accel(self, char_accel: float) -> float:
        """Calculate the mass for this sail to give a characteristic acceleration."""
        return (
            SOLAR_RADIATION_PRESSURE_P0 * (self.G + self.K) * self.area_m2 / char_accel
        )


@njit
def wright_sail(area: float) -> SailProperties:
    """Returns a sail with optical properties as given by Wright, and given area."""
    return SailProperties(area, 0.88, 0.94, 0.79, 0.55, 0.05, 0.55)


@njit
def ideal_sail(area: float) -> SailProperties:
    """Returns a sail with theoretical ideal optical propeties, and given area."""
    return SailProperties(area, 1, 1, 0, 0, 1, 1)


@njit
def null_sail() -> SailProperties:
    """Return a sail that does nothing."""
    return ideal_sail(0)
