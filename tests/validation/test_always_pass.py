"""
Validation test that always passes for CI's sake.
"""
from ..common_test_utils import TestCase


class AlwaysPass(TestCase):
    def test_always_pass(self):
        self.assertTrue(True)
