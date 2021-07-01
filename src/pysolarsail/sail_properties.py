"""
Stores and handles sail optical properties and generates the sail properties object.
"""
from collections import OrderedDict

import numpy as np
from numba import float64, jit
from numba.experimental import jitclass

SAIL_PROPERTIES_SPEC = OrderedDict(
    [
        ("G", float64),
        ("H", float64),
        ("K", float64),
    ]
)


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

    def calculate_force(
        self,
        solar_radiation_pressure: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Calculate the force on a solar sail in the e_r frame."""
        delta = self.delta(beta)
        return (
            solar_radiation_pressure
            * self.area_m2
            * self.Q(beta)
            * np.array(
                [
                    np.cos(delta),
                    np.sin(delta) * np.cos(alpha),
                    np.sin(delta) * np.sin(alpha),
                ]
            )
        )


@jit
def wright_sail(area: float) -> SailProperties:
    """Returns a sail with optical properties as given by Wright, and given area."""
    return SailProperties(area, 0.88, 0.94, 0.79, 0.55, 0.05, 0.55)


@jit
def ideal_sail(area: float) -> SailProperties:
    """Returns a sail with theoretical ideal optical propeties, and given area."""
    return SailProperties(area, 1, 1, 0, 1, 1, 1)
