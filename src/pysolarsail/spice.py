"""
Interactions with SPICE.
"""
import logging
from contextlib import ContextDecorator
from datetime import datetime
from os import chdir, getcwd
from os.path import dirname
from types import TracebackType
from typing import Any, Callable, Iterable, Union

import numpy as np
import spiceypy

# Convenience type definitions.
StrOrIterStr = Union[str, Iterable[str]]

# Default frame should be J2000 (which implicitly makes everything relative to the
# ICRF), with no corrections, and we want to calculate positions to the solar system
# barycenter.
FRAME = "J2000"
CORRECTION = "NONE"
REFERENCE = "SOLAR SYSTEM BARYCENTER"


def _in_spice_dir(func: Callable) -> Callable:
    """Decorator used in the SpiceKernel context manager to load in a given folder."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Move to SPICE_FOLDER and run func()."""
        orig_dir = getcwd()
        chdir(args[0].kernel_dir)  # Because args[0] should always be the instance.
        output = func(*args, **kwargs)
        chdir(orig_dir)
        return output

    return wrapper


class SpiceKernel(ContextDecorator):
    """Context manager for having spice loaded."""

    def __init__(self, kernel_file: str) -> None:
        self.kernel_file = kernel_file
        self.kernel_dir = dirname(self.kernel_file)

    @_in_spice_dir
    def __enter__(self) -> None:
        """Move to spice directory, load spice kernel."""
        spiceypy.furnsh(self.kernel_file)
        logging.info(
            f"Loaded {self.kernel_file}, {spiceypy.ktotal('ALL')} kernels now loaded."
        )

    @_in_spice_dir
    def __exit__(self, exc, exca, exc_trace: TracebackType) -> None:
        spiceypy.unload(self.kernel_file)
        logging.info(
            f"Unloaded {self.kernel_file}, {spiceypy.ktotal('ALL')} kernels still "
            f"loaded"
        )


def get_pos(name: str, times: Union[float, np.ndarray]) -> np.ndarray:
    """Get the position of a planet from spice, in km."""
    pos, _ = spiceypy.spkpos(name, times, FRAME, CORRECTION, REFERENCE)
    return pos


def get_vel(name: str, times: Union[float, np.ndarray]) -> np.ndarray:
    """Get the velocity of a planet from spice, in km/s."""
    state, _ = spiceypy.spkezr(name, times, FRAME, CORRECTION, REFERENCE)
    if type(state) is list:
        vel = np.vstack(state)[:, 3:6]
    else:
        vel = state[3:6]
    return vel


def get_eph_time(time: datetime) -> float:
    """Convert a python datetime to an ephemeris time."""
    return spiceypy.str2et(time.isoformat())


def get_mean_radius(name: str) -> float:
    """Return the mean radius of a given planet."""
    return np.mean(spiceypy.bodvrd(name, "RADII", maxn=3)[1])


def get_gravity(name: str) -> float:
    """Return the gravitation (G * M) for a body."""
    return spiceypy.bodvrd(name, "GM", maxn=1)[1][0]


# Useful stuff = https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/index.html
# clight() returns the speed of light in km/s
# j2000() returns the time of j2000
# spd() returns seconds per day
