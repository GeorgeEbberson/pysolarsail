"""
The basic spacecraft class.
"""
from datetime import datetime
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import float64, njit
from numba.experimental import jitclass
from numba.typed import List

from pysolarsail.bodies import SpiceBody
from pysolarsail.numba_utils import class_spec
from pysolarsail.sail_properties import SailProperties, null_sail, wright_sail
from pysolarsail.spice import SpiceKernel, get_eph_time
from pysolarsail.units import M_PER_AU, SPEED_OF_LIGHT_M_S

# These coefficients are taken from
# https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf
# repeated in the Butcher tableau on this page:
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
RKF_H_COEFFS = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2], dtype=np.float64)
_RKF_K_COEFFS = np.array(
    [
        [          0,            0,            0,           0,        0, 0],
        [      1 / 4,            0,            0,           0,        0, 0],
        [     3 / 32,       9 / 32,            0,           0,        0, 0],
        [1932 / 2197, -7200 / 2197,  7296 / 2197,           0,        0, 0],
        [  439 / 216,           -8,   3680 / 513, -845 / 4104,        0, 0],
        [    -8 / 27,            2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
    ],
    dtype=np.float64,
)
_RKF_YPLUS_COEFFS = np.array(
    [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0],
    dtype=np.float64,
)
_RKF_ZPLUS_COEFFS = np.array(
    [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
    dtype=np.float64,
)

# Now we have to precompute each of the above as a 4-D equivalent because np.newaxis is
# unsupported.
RKF_K_COEFFS = np.zeros((6, 6, 1, 1))
RKF_YPLUS_COEFFS = np.zeros((6, 1, 1))
RKF_ZPLUS_COEFFS = np.zeros((6, 1, 1))
for idx in range(6):
    RKF_K_COEFFS[idx, :, :, :] = np.expand_dims(_RKF_K_COEFFS[idx], axis=(1, 2))
    RKF_YPLUS_COEFFS[idx, :, :] = _RKF_YPLUS_COEFFS[idx]
    RKF_ZPLUS_COEFFS[idx, :, :] = _RKF_ZPLUS_COEFFS[idx]


SOLAR_SAILCRAFT_SPEC = [
    ("pos_m", float64[::1]),
    ("vel_m_s", float64[::1]),
    ("sail", class_spec(SailProperties)),
    ("alpha_rad", float64),
    ("beta_rad", float64),
    ("mass", float64),
]


@jitclass(SOLAR_SAILCRAFT_SPEC)
class SolarSailcraft(object):
    """The base class for a solar sail spacecraft."""

    def __init__(self, pos: np.ndarray, vel: np.ndarray, sail: SailProperties) -> None:
        """Create a new class instance."""
        self.pos_m = pos
        self.vel_m_s = vel
        self.sail = sail
        self.alpha_rad = 0
        self.beta_rad = 0
        self.mass = 1

    def set_time(self, time: float):
        return None


@njit
def unit_vector_and_mag(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return a unit vector and the magnitude of vec."""
    mag = np.linalg.norm(vec)
    return np.divide(vec, mag), mag


@njit
def solarsail_acceleration(
    craft: SolarSailcraft,
    bodies: Iterable[SpiceBody],
) -> np.ndarray:
    """Calculate the acceleration on a solar sailcraft given some planets and stars."""
    accel = np.zeros((3,), dtype=np.float64)
    for body in bodies:
        craft_to_body_unit_vec, rad = unit_vector_and_mag(body.pos_m - craft.pos_m)
        sail_contrib = (
            (
                body.radiation_w_m2
                * (M_PER_AU ** 2)
                * craft.sail.aqf(craft.alpha_rad, craft.beta_rad)
                / (SPEED_OF_LIGHT_M_S * craft.mass)
            )
            if body.is_star
            else np.array([0, 0, 0], dtype=np.float64)
        )
        accel += (1 / rad ** 2) * (
            (body.gravitation_parameter_m3_s2 * craft_to_body_unit_vec) + sail_contrib
        )
    return accel


@njit
def rkf_rhs(
    time: float,
    state: Tuple[np.array],
    bodies: Iterable[SpiceBody],
    craft: SolarSailcraft,
) -> Tuple[np.ndarray, np.ndarray]:
    """The righthand side of the Runge-Kutta-Fehlberg equation of motion.

    This should return a 2-vector of the velocity vector and the acceleration vector at
    time, i.e. should be a 2-vector of 3-vectors.
    """

    for obj in bodies:
        obj.set_time(time)
    craft.set_time(time)

    return np.vstack((state[1], solarsail_acceleration(craft, bodies)))


@njit
def get_init_state(craft: SolarSailcraft) -> np.ndarray:
    """Get the initial state."""
    return np.vstack((craft.pos_m, craft.vel_m_s))


@njit
def compute_k(time, X, timestep, model, craft):
    """Calculate yk+1 and zk+1."""
    times = time + RKF_H_COEFFS * timestep
    k = np.zeros((6, 2, 3))
    for idx in range(6):
        k[idx, :, :] = timestep * rkf_rhs(
            times[idx],
            X + np.sum(RKF_K_COEFFS[idx] * k, axis=0),
            model,
            craft,
        )
    return k


@njit
def find_optimal_stepsize(t, X, dt, model, craft, tol):
    """Return the optimal stepsize."""
    k = compute_k(t, X, dt, model, craft)
    y_kplus1 = X + np.sum(RKF_YPLUS_COEFFS * k, axis=0)
    z_kplus1 = X + np.sum(RKF_ZPLUS_COEFFS * k, axis=0)

    # Compare the two to find the optimal stepsize.
    s = np.power((tol * dt) / (2 * np.max(np.abs(z_kplus1 - y_kplus1))), 0.25)
    return np.min([s * dt, 86400])


@njit
def solve_rkf(craft, start_time, end_time, model, init_time_step=86400, tol=1e-7):
    """Run the RKF algorithm."""
    dt = init_time_step
    X = get_init_state(craft)
    t = start_time

    n_cols = 8 + 3 * len(model)
    results_list = [[0] * n_cols]

    while t < end_time:

        real_time_step = find_optimal_stepsize(t, X, dt, model, craft, tol)

        # Finally find the real k and find the real state.
        k_real = compute_k(t, X, real_time_step, model, craft)
        X += np.sum(RKF_YPLUS_COEFFS * k_real, axis=0)

        # Write the output data.
        results = [
            t,
            real_time_step,
            X[0, 0], X[0, 1], X[0, 2],
            X[1, 0], X[1, 1], X[1, 2]
        ]
        positions = sum([body.pos_m.tolist() for body in model], [])
        results_list.append(results + positions)

        # Increment time for next timestep.
        t += real_time_step
        dt = real_time_step

    return np.array(results_list)


if __name__ == "__main__":
    with SpiceKernel(r"D:\dev\solarsail\src\solarsail\spice_files\metakernel.txt"):
        start_time = get_eph_time(np.datetime64(datetime(2003, 1, 15, 12)))
        end_time = get_eph_time(np.datetime64(datetime(2004, 8, 11, 12)))

        sun = SpiceBody("Sun", start_time, end_time, radiation=1368)
        mercury = SpiceBody("Mercury", start_time, end_time)
        venus = SpiceBody("Venus", start_time, end_time)
        earth = SpiceBody("Earth", start_time, end_time)
        mars = SpiceBody("Mars", start_time, end_time)
        jupiter = SpiceBody("Jupiter", start_time, end_time)
        saturn = SpiceBody("Saturn", start_time, end_time)
        uranus = SpiceBody("Uranus", start_time, end_time)
        neptune = SpiceBody("Neptune", start_time, end_time)
        pluto = SpiceBody("Pluto", start_time, end_time)
        planets = List()  # numba typed list avoids reflected list problems.
        planets.append(sun)
        planets.append(mercury)
        planets.append(venus)
        planets.append(earth)
        planets.append(mars)
        planets.append(jupiter)
        planets.append(saturn)
        planets.append(uranus)
        planets.append(neptune)
        planets.append(pluto)

        craft = SolarSailcraft(
            earth.pos_m + np.array([8_000_000, 0, 0]),
            earth.vel_m_s,
            # wright_sail(float(800 * 800)),
            null_sail(),
        )

        a = solve_rkf(craft, start_time, end_time, planets)
        print(a)
        fig = plt.figure()
        plt.plot(a[:, 2] / M_PER_AU, a[:, 3] / M_PER_AU)
        plt.plot(a[:, 11] / M_PER_AU, a[:, 12] / M_PER_AU)
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.axis("equal")

        fig2 = plt.figure()
        plt.plot(a[:, 1], label="real_time_step")
        # plt.plot(a[:, 0], a[:, 2], label="X[0,0]")
        # plt.plot(a[:, 0], a[:, 3], label="X[0,1]")
        # plt.plot(a[:, 0], a[:, 4], label="X[0,2]")
        # plt.plot(a[:, 0], a[:, 5], label="X[1,0]")
        # plt.plot(a[:, 0], a[:, 6], label="X[1,1]")
        # plt.plot(a[:, 0], a[:, 7], label="X[1,2]")
        plt.legend()

        plt.show()
