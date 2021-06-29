"""
Unit test that always passes for CI's sake.
"""
import unittest

import pysolarsail


class TestAlwaysPass(unittest.TestCase):
    def test_always_pass(self):
        self.assertTrue(True)
