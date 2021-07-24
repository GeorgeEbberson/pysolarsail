"""
Solar system bodies.
"""
from datetime import datetime

import numpy as np
from numba import float64, types
from numba.experimental import jitclass

from pysolarsail.spice import SpiceKernel, get_gravity, get_pos, get_vel

SPICE_BODY_SPEC = [
    ("name", types.string),
    ("pos_m", float64[::1]),
    ("vel_m_s", float64[::1]),
    ("gravitation_parameter_m3_s2", float64),
    ("radiation_w_m2", float64),
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
    ) -> None:
        """Initialise the instance, and check that spice has the correct data
        loaded (i.e. there are kernels loaded for the correct times).
        """
        self.name = name
        self.pos_m = np.empty((3,), dtype=np.float64)
        self.vel_m_s = np.empty((3,), dtype=np.float64)
        self.gravitation_parameter_m3_s2 = get_gravity(name)
        self.radiation_w_m2 = radiation

        # Initialise the values to the start time.
        self.set_time(start_eph_time)

    def set_time(self, eph_time: float) -> None:
        self.pos_m = get_pos(self.name, eph_time)
        self.vel_m_s = get_vel(self.name, eph_time)

    @property
    def is_star(self) -> bool:
        """True if the body is a star."""
        return not self.radiation_w_m2 == 0.0


if __name__ == "__main__":
    with SpiceKernel(r"D:\dev\solarsail\src\solarsail\spice_files\metakernel.txt"):
        s = SpiceBody(
            "Sun",
            np.datetime64(datetime(2020, 1, 1)),
            np.datetime64(datetime(2021, 1, 1)),
            radiation=1368,
        )
        print(s)
        print(s.radiation_w_m2)
        print(s.gravitation_parameter_m3_s2)
        print(s.is_star)
