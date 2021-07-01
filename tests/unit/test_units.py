"""
Test the units module from pysolarsail.
"""
import re
import unittest
from random import random

import pysolarsail.units
from pysolarsail.units import M_PER_AU, au_to_m, m_to_au

# Regex which matches all constants.
CONSTANT_REGEX = re.compile("[A-Z][A-Z0-9_]+")


class TestConstants(unittest.TestCase):
    """Ensure that all constants exist and are floats."""

    def test_constants_are_floats(self):
        """Gets the constants and checks the type of each is a float."""

        for name in dir(pysolarsail.units):
            if CONSTANT_REGEX.match(name):
                self.assertIsInstance(getattr(pysolarsail.units, name), float)


class TestConversions(unittest.TestCase):
    """Very simple conversion tests."""

    def test_m_to_au(self):
        multiple = random()
        self.assertEqual(m_to_au(multiple * M_PER_AU), multiple)

    def test_au_to_m(self):
        multiple = random()
        self.assertEqual(au_to_m(multiple / M_PER_AU), multiple)
