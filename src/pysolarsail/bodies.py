"""
Solar system bodies.
"""
import numpy as np
from numba import boolean, float64, types
from numba.experimental import jitclass

from pysolarsail.spice import get_gravity, get_pos, get_vel

SPICE_BODY_SPEC = [
    ("name", types.string),
    ("pos_m", float64[::1]),
    ("vel_m_s", float64[::1]),
    ("gravitation_parameter_m3_s2", float64),
    ("radiation_w_m2", float64),
    ("gravity", boolean),
]


@jitclass(SPICE_BODY_SPEC)
class SpiceBody(object):
    """Base class for a body which gets its position from SPICE."""

    def __init__(
        self,
        name: str,
        start_eph_time: float,
        end_eph_time: float,
        radiation: float = 0,
        gravity: bool = True,
    ) -> None:
        """Initialise the instance, and check that spice has the correct data
        loaded (i.e. there are kernels loaded for the correct times).
        """
        self.name = name
        self.pos_m = np.empty((3,), dtype=np.float64)
        self.vel_m_s = np.empty((3,), dtype=np.float64)
        self.gravitation_parameter_m3_s2 = get_gravity(name)
        self.radiation_w_m2 = radiation
        self.gravity = gravity

        # Initialise the values to the start time.
        self.set_time(start_eph_time)

    def set_time(self, eph_time: float) -> None:
        self.pos_m = get_pos(self.name, eph_time)
        self.vel_m_s = get_vel(self.name, eph_time)

    @property
    def is_star(self) -> bool:
        """True if the body is a star."""
        return not self.radiation_w_m2 == 0.0
