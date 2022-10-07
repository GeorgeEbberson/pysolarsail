"""
Interactions with SPICE.
"""
import logging
from contextlib import ContextDecorator, contextmanager
from datetime import datetime
from os import chdir
from pathlib import Path
from types import TracebackType
from typing import Any, Generator, Iterable, Union, cast

import numpy as np
import spiceypy
from numba import njit, objmode

# Default frame should be J2000 (which implicitly makes everything relative to the
# ICRF), with no corrections, and we want to calculate positions to the solar system
# barycenter.
FRAME = "J2000"
CORRECTION = "NONE"
REFERENCE = "SOLAR SYSTEM BARYCENTER"


@contextmanager
def _change_dir(change: bool, target_dir: Path) -> Generator[None, None, None]:
    """Change directory to target_dir if change is True."""
    orig_dir = Path.cwd()
    try:
        if change:
            chdir(target_dir)
        yield

    finally:
        if change:
            chdir(orig_dir)


def _make_path(file_string: str) -> Path:
    """Convert a kernel filename string to a Path object and check it is valid."""
    path = Path(file_string)
    if not path.is_file():
        raise FileNotFoundError(f"File {path.name} is not a valid file.")
    return path


class SpiceKernel(ContextDecorator):
    """Context manager for loading and unloading SPICE kernels.

    Loads the kernel(s) given using `spiceypy.furnsh()` and unloads them using
    `spiceypy.unload()` when the context is exited. Can also be called as a decorator
    around a function, which treats the entire function as being in context (i.e. the
    entire function is executed with the kernels loaded).

    :param kernel_files: A kernel file or list of kernel files to load.
    :param allow_change_dir: Whether to change folder to load kernels. Defaults true.
    """

    def __init__(
        self,
        kernel_files: Union[str, Iterable[str]],
        allow_change_dir: bool = True,
    ) -> None:
        """Initialise the context and check all given files are valid."""

        # If we've been given a single string, make it a list so we can just work on
        # iterable sequences in the class.
        if isinstance(kernel_files, str):
            kernel_files = [kernel_files]

        # Now check each string in the list is a file, and convert to a list of pathlib
        # Path() objects.
        self.kernel_file_list = [_make_path(x) for x in kernel_files]
        self.allow_change_dir = allow_change_dir

    def __enter__(self) -> None:
        """Move to the directory of each kernel, then load it using `furnsh`."""
        for kernel in self.kernel_file_list:
            with _change_dir(self.allow_change_dir, kernel.parent):
                spiceypy.furnsh(str(kernel))
                logging.info(
                    f"Loaded {kernel.name}, {spiceypy.ktotal('ALL')} kernels now "
                    f"loaded."
                )

    def __exit__(self, exc: Any, exca: Any, excb: TracebackType) -> None:
        for kernel in self.kernel_file_list:
            with _change_dir(self.allow_change_dir, kernel.parent):
                spiceypy.unload(str(kernel))
                logging.info(
                    f"Unloaded {kernel.name}, {spiceypy.ktotal('ALL')} kernels still "
                    f"loaded"
                )


@njit
def get_pos(name: str, eph_time: float) -> np.ndarray:
    """Get the position of a planet from spice, in km."""
    with objmode(pos="float64[::1]"):
        pos, _ = spiceypy.spkpos(name, eph_time, FRAME, CORRECTION, REFERENCE)
    return pos * 1000


@njit
def get_vel(name: str, eph_time: float) -> np.ndarray:
    """Get the velocity of a planet from spice, in km/s."""
    with objmode(vel="float64[::1]"):
        state, _ = spiceypy.spkezr(name, eph_time, FRAME, CORRECTION, REFERENCE)
        vel = cast(np.ndarray, state[3:6])
    return vel * 1000


@njit
def get_eph_time(time: np.datetime64) -> float:
    """Convert a python datetime to an ephemeris time."""
    with objmode(et="float64"):
        et = spiceypy.str2et(np.datetime_as_string(time))
    return et


@njit
def get_mean_radius(name: str) -> float:
    """Return the mean radius of a given planet."""
    with objmode(rad="float64[::1]"):
        _, rad = spiceypy.bodvrd(name, "RADII", maxn=3)
    return cast(float, np.mean(rad))


@njit
def get_gravity(name: str) -> float:
    """Return the gravitational parameter (G * M) for a body."""
    with objmode(gm="float64"):
        gm = cast(float, spiceypy.bodvrd(name, "GM", maxn=1)[1])
    return gm * (1000**3)  # SPICE uses km^3 / s^2


def check_kernels_loaded(
    names: Iterable[str],
    start_date: datetime,
    end_date: datetime,
) -> bool:
    """Check that kernel data is loaded for the given body between the given times.

    Iterates over the bodies one by one, using the underlying SPICE functions to
    check whether kernel data is loaded for the time covered. In the event that
    multiple kernels cover the same time window, the discovery of the first
    matching window causes True.

    :param names: An iterable of strings of body names.
    :param start_date: A datetime representing the start of the window.
    :param end_data: A datetime representing the end of the window.
    """
    window = (get_eph_time(start_date), get_eph_time(end_date))

    return all(
        [
            is_kernel_loaded_for_one_body(name, window)
            for name in names
        ]
    )


def is_kernel_loaded_for_one_body(name, window):
    """Check if a kernel is loaded covering one body."""
    for file in loaded_kernel_files("SPK"):
        if win := spiceypy.spkcov(file, spiceypy.bods2c(name)):
            for interval in intervals_in_window(win):
                if interval_contains_interval(interval, window):
                    return True
    return False


def intervals_in_window(window):
    """Generator of the intervals within a window."""
    for idx in range(spiceypy.wncard(window)):
        yield spiceypy.wnfetd(window, idx)


def loaded_kernel_files(kernel_type):
    """Generator of the files that are currently loaded."""
    for idx in range(spiceypy.ktotal(kernel_type)):
        k_name, k_type, k_source, _ = spiceypy.kdata(idx, kernel_type)
        assert k_type == kernel_type, f"Kernel type {k_type} does not match {kernel_type}"
        yield str(Path(k_source).parent / k_name)  # By convention we give kernels relative to their meta.


def interval_contains_interval(container, contained):
    """Return True if contained lies wholly within container."""
    return container[0] <= contained[0] < contained[1] <= container[1]  # since windows should be sorted.


# Useful stuff = https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/index.html
# j2000() returns the time of j2000
