"""
Tests for the spiceypy interface functions.
"""
from os import chdir
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spiceypy

from pysolarsail.spice import (
    get_eph_time,
    get_gravity,
    get_mean_radius,
    get_pos,
    get_vel,
)

from ..common_test_utils import TestCase

NAT = np.datetime64()
J2000 = np.datetime64("2000-01-01")
SUN = "SUN"
DISCARDED = np.inf
STRING = "I'm a string!"


class TestSpiceUnit(TestCase):
    """Unit tests for the spice-accessing functions (spiceypy mocked)."""

    def test_get_pos(self) -> None:
        """Check that only the first return value is kept and that spkpos is called
        only once."""
        with patch("spiceypy.spkpos") as spkpos, patch("spiceypy.str2et") as str2et:
            ret_val = (np.array([0, 0, 0]), DISCARDED)
            spkpos.return_value = ret_val
            self.assertArrayEqual(get_pos(SUN, NAT), ret_val[0])
            spkpos.assert_called_once()
            self.assertEqual(len(spkpos.call_args[0]), 5)
            str2et.assert_called_once()

    def test_get_vel(self) -> None:
        """Check that spkezr is called once and that the output array is always
        N-by-3, regardless of input length."""
        with patch("spiceypy.spkezr") as spkezr, patch("spiceypy.str2et") as str2et:
            spkezr.return_value = (
                np.array([DISCARDED, DISCARDED, DISCARDED, 1, 2, 3]),
                DISCARDED,
            )
            result = get_vel(SUN, NAT)
            self.assertEqual(type(result), np.ndarray)
            self.assertEqual(result.shape[-1], 3)
            spkezr.assert_called_once()
            self.assertEqual(len(spkezr.call_args[0]), 5)
            str2et.assert_called_once()

    def test_get_eph_time(self) -> None:
        """Simple check that the isoformat method is called (i.e. we've cast to a
        string)."""
        test_time = np.datetime64("1970-01-01")
        with patch("spiceypy.str2et") as str2et:
            str2et.return_value = 1.2
            result = get_eph_time(test_time)
            self.assertEqual(result, 1.2)
            self.assertTrue(str2et.called)
            self.assertEqual(type(str2et.call_args[0][0]), np.str_)

    def test_get_mean_radius(self) -> None:
        """Check that mean radius returns a single value and ignores the 'num'
        return."""
        with patch("spiceypy.bodvrd") as bodvrd:
            bodvrd.return_value = (DISCARDED, np.array([1, 2, 3]))
            result = get_mean_radius(SUN)
            self.assertEqual(bodvrd.call_args[0][1], "RADII")
            self.assertEqual(result, float(2))

    def test_get_gravity(self) -> None:
        """Check that a single value is returned and it's properly indexed."""
        with patch("spiceypy.bodvrd") as bodvrd:
            bodvrd.return_value = (DISCARDED, np.array([1]))
            result = get_gravity(SUN)
            self.assertEqual(bodvrd.call_args[0][1], "GM")
            self.assertEqual(result, float(1))


class TestSpiceIntegration(TestCase):
    """More integration-y tests for spice-accessing functions (spiceypy not mocked)."""

    _metakernel = None
    _exec_dir = None

    @classmethod
    def setUpClass(cls) -> None:
        """Load the kernels for the test."""
        cls._metakernel = Path(__file__).parent / "kernels" / "metakernel.txt"
        cls._exec_dir = Path.cwd()
        chdir(cls._metakernel.parent)
        spiceypy.furnsh(str(cls._metakernel))
        chdir(cls._exec_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        """Unload the kernels."""
        spiceypy.unload(str(cls._metakernel))

    def test_get_pos(self) -> None:
        """Test that get pos returns an array with correct shape."""
        pos = get_pos(SUN, J2000)
        self.assertEqual(pos.shape, (3,))
