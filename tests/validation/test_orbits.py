"""
Sanity checks to make sure things behave obviously.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pysolarsail.bodies import SpiceBody
from pysolarsail.sail_properties import null_sail
from pysolarsail.spacecraft import SolarSailcraft, solve_rkf
from pysolarsail.spice import SpiceKernel, get_eph_time
from pysolarsail.units import CONSTANT_OF_GRAVITATION_M3_KG_S2, M_PER_AU
from tests.common_test_utils import KERNELS_DIR, TestCase, cases
from tests.validation.test_dachwald import (
    DACHWALD_MERCURY_ALPHA_FILE,
    DACHWALD_MERCURY_BETA_FILE,
    DACHWALD_MERCURY_KERNELS,
)


@pytest.mark.validation
class TestOrbits(TestCase):
    """Sanity checks to try and find the problem.

    Uses Dachwald's kernels and setup as it doesn't actually matter.
    """

    @cases(
        ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Pluto"]
    )
    def test_planet_follows_orbit(self, name):
        """Simulate a planet and make sure it's close to the actual planet."""
        with SpiceKernel(DACHWALD_MERCURY_KERNELS, root=KERNELS_DIR):
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
                str(DACHWALD_MERCURY_ALPHA_FILE),
                str(DACHWALD_MERCURY_BETA_FILE),
                start_time,
                mass=(
                    planet.gravitation_parameter_m3_s2
                    / CONSTANT_OF_GRAVITATION_M3_KG_S2
                ),
            )

            results = solve_rkf(craft, start_time, end_time, [sun, planet])

        # Calculate things we care about, but don't assert yet in case
        # we want to draw plots.

        # Distance from craft (simulated Earth) to Earth calculated by Spice.
        # Since they used the same time step should be in the same place.
        planet_idx = [
            f"{name.lower()}_pos_x",
            f"{name.lower()}_pos_y",
            f"{name.lower()}_pos_z",
        ]
        craft_idx = ["craft_pos_x", "craft_pos_y", "craft_pos_z"]
        planet_pos = results[planet_idx].values
        craft_pos = results[craft_idx].values
        dists = np.linalg.norm(planet_pos - craft_pos, axis=1) / M_PER_AU

        if self.SHOULD_PLOT:
            # Plot positions of
            fig = plt.figure()  # noqa: F841
            plt.plot(results[craft_idx[0]], results[craft_idx[1]])
            plt.plot(results[planet_idx[0]], results[planet_idx[1]])
            ax = plt.gca()
            ax.axis("equal")

            fig2 = plt.figure()  # noqa: F841
            plt.plot(results["time"], results["craft_vel_x"], ":b", label="velx")
            plt.plot(results["time"], results["craft_vel_y"], ":g", label="vely")
            plt.plot(
                results["time"],
                results[f"{name.lower()}_vel_x"],
                ":k",
                label="planetvelx",
            )
            plt.plot(
                results["time"],
                results[f"{name.lower()}_vel_y"],
                ":r",
                label="planetvely",
            )
            plt.legend()

            fig3 = plt.figure()  # noqa: F841
            plt.plot(results["time"], dists, "-k", label="sim to spice dist")
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
        # self.assertLessEqual(threshold - np.max(dists), 0.001)
