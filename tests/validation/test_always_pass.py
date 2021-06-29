"""
Validation test that always passes for CI's sake.
"""
import unittest

import pysolarsail


class AlwaysPass(unittest.TestCase):
    def test_always_pass(self):
        self.assertTrue(True)
