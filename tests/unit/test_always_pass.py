"""
Unit test that always passes for CI's sake.
"""
import pytest

from ..common_test_utils import TestCase


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.validation
class TestAlwaysPass(TestCase):
    def test_always_pass(self):
        self.assertTrue(True)
