"""
Sanity checks to make sure things behave obviously.
"""
from datetime import datetime

import pytest
import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized

from pysolarsail.bodies import SpiceBody
from pysolarsail.sail_properties import null_sail
from pysolarsail.spacecraft import SolarSailcraft, solve_rkf
from pysolarsail.spice import (
    SpiceKernel,
    get_eph_time,
)
from pysolarsail.units import M_PER_AU, CONSTANT_OF_GRAVITATION_M3_KG_S2

from ..common_test_utils import TestCase

DEBUG_PLOTS = False

@pytest.mark.validation
class TestOrbits(TestCase):
    """Sanity checks to try and find the problem."""

    @parameterized.expand([
        ("Mercury", ),
        ("Venus", ),
        ("Earth", ),
        ("Mars", ),
        ("Jupiter"),
        ("Saturn", ),
        ("Uranus", ),
        ("Pluto", ),
        ])
    def test_planet_follows_orbit(self, name):
        """Simulate a planet and make sure it's close to the actual planet."""

        with SpiceKernel(r"D:\dev\solarsail\src\solarsail\spice_files\metakernel.txt"):
            # Simulate the year 2020.
            start_time = get_eph_time(np.datetime64(datetime(2020, 1, 1, 12)))
            end_time = get_eph_time(np.datetime64(datetime(2021, 1, 1, 12)))

            # Get Earth and the Sun as reference.
            # Sun has no radiation so we're just simulating gravity.
            # Earth has no gravity so it doesn't affect the craft.
            sun = SpiceBody("Sun", start_time, end_time, radiation=0)
            planet = SpiceBody(name, start_time, end_time, gravity=False)

            # Make a craft that matches Earth.
            craft = SolarSailcraft(
                planet.pos_m,
                planet.vel_m_s,
                null_sail(),
                r"D:\dev\solarsail\data\dachwald_mercury_alpha.csv",
                r"D:\dev\solarsail\data\dachwald_mercury_beta.csv",
                start_time,
                mass=(planet.gravitation_parameter_m3_s2 / CONSTANT_OF_GRAVITATION_M3_KG_S2),
            )

            results = solve_rkf(craft, start_time, end_time, [sun, planet])

        # Calculate things we care about, but don't assert yet in case
        # we want to draw plots.

        # Distance from craft (simulated Earth) to Earth calculated by Spice.
        # Since they used the same time step should be in the same place.
        dists = np.linalg.norm(results[:, 25:28] - results[:, 2:5], axis=1) / M_PER_AU

        if DEBUG_PLOTS:
            # Plot positions of
            fig = plt.figure()
            plt.plot(results[:, 2], results[:, 3])
            plt.plot(results[:, 25], results[:, 26])
            ax = plt.gca()
            ax.axis("equal")

            fig2 = plt.figure()
            plt.plot(results[:, 0], results[:, 5], ":b", label="velx")
            plt.plot(results[:, 0], results[:, 6], ":g", label="vely")
            plt.plot(results[:, 0], results[:, 31], ":k", label="planetvelx")
            plt.plot(results[:, 0], results[:, 32], ":r", label="planetvely")
            plt.legend()

            fig3 = plt.figure()
            plt.plot(results[:, 0], dists, "-k", label="sim to spice dist")
            plt.ylabel("Distance (AU)")
            plt.legend()

            plt.show()

        # The positions of the planet and the spacecraft should match closely.
        # 0.003 AU is what we can achieve for now, hopefully will improve in time.
        threshold = 0.0075
        self.assertLessEqual(np.max(dists), threshold)

        # We also want to fail if the achieved distance from sim to spice is
        # much lower than the threshold, so we can ratchet towards 0 without
        # regressing.
        #self.assertLessEqual(threshold - np.max(dists), 0.001)