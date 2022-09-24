"""
Validation test that we can achieve the same results as Dachwald.
"""
from datetime import datetime

import pytest
import matplotlib.pyplot as plt
import numpy as np

from pysolarsail.bodies import SpiceBody
from pysolarsail.sail_properties import ideal_sail
from pysolarsail.spacecraft import SolarSailcraft, solve_rkf
from pysolarsail.spice import SpiceKernel, get_eph_time
from pysolarsail.units import M_PER_AU, SOLAR_IRRADIANCE_W_M2

from ..common_test_utils import TestCase

@pytest.mark.validation
class TestDachwald(TestCase):
    """Tests from Dachwald's cases."""

    def test_dachwald_mercury(self):
        """Test that we achieve Dachwald's Mercury Rendezvous."""
        with SpiceKernel(r"D:\dev\solarsail\src\solarsail\spice_files\metakernel.txt"):
            start_time = get_eph_time(np.datetime64(datetime(2003, 1, 15, 12)))
            end_time = get_eph_time(np.datetime64(datetime(2004, 8, 11, 12)))

            sun = SpiceBody("Sun", start_time, end_time, radiation=SOLAR_IRRADIANCE_W_M2)
            mercury = SpiceBody("Mercury", start_time, end_time, gravity=False)
            earth = SpiceBody("Earth", start_time, end_time, gravity=False)

            planets = [sun, mercury, earth]

            craft = SolarSailcraft(
                earth.pos_m,
                earth.vel_m_s,
                ideal_sail(125 * 125),
                r"D:\dev\solarsail\data\dachwald_mercury_alpha.csv",
                r"D:\dev\solarsail\data\dachwald_mercury_beta.csv",
                start_time,
                char_accel=0.00055,  # 0.55 mm/s^2
            )

            results = solve_rkf(craft, start_time, end_time, planets)
            if self.SHOULD_PLOT:
                fig = plt.figure()
                plt.title("rkf")
                plt.plot(results[:, 2] / M_PER_AU, results[:, 3] / M_PER_AU, "-k")
                plt.plot(results[:, 25] / M_PER_AU, results[:, 26] / M_PER_AU, "-r")
                ax = plt.gca()
                ax.spines['top'].set_color('none')
                ax.spines['left'].set_position('zero')
                ax.spines['right'].set_color('none')
                ax.spines['bottom'].set_position('zero')
                ax.axis("equal")
                plt.show()

            # Now make assertions. Dachwald says:
            #  * Final distance to Mercury approx 57000 km
            #  * Final relative velocity approx 57 m/s
            # final_dist_mercury_to_sail = np.linalg.norm()
            # final_rel_vel_mercury_to_sail = np.linalg.norm()
            # self.assertAlmostEqual(final_dist_mercury_to_sail, 57E6, delta=1E6)
            # self.assertAlmostEqual(final_rel_vel_mercury_to_sail, 57, delta=1)

