# -*- coding: utf-8 -*-
"""Test bandit interface implementation."""
import pytest

import logging

from moe.bandit.bandit_interface import BanditInterface
from moe.tests.bandit.bandit_test_case import BanditTestCase


class TestBanditInterface(BanditTestCase):

    """Verify that different historical infos return correct results."""

    @pytest.fixture(autouse=True)
    def disable_logging(self):
        """Disable logging (for the duration of this test case)."""
        logging.disable(logging.CRITICAL)

        def fin():
            """Re-enable logging (so other test cases are unaffected)."""
            logging.disable(logging.NOTSET)

    def test_empty_arm_invalid(self):
        """Test empty ``arms_to_allocations`` causes an ValueError."""
        with pytest.raises(ValueError):
            BanditInterface.choose_arm({})

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm."""
        arms_to_allocations = {"arm1": 1.0}
        assert BanditInterface.choose_arm(arms_to_allocations) == "arm1"

    def test_two_arms_one_winner(self):
        """Check that the two-arms case with one winner always returns the winning arm."""
        arms_to_allocations = {"arm1": 1.0, "arm2": 0.0}
        assert BanditInterface.choose_arm(arms_to_allocations) == "arm1"

    def test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return one of the two winners."""
        arms_to_allocations = {"arm1": 0.5, "arm2": 0.5, "arm3": 0.0}
        assert BanditInterface.choose_arm(arms_to_allocations) in frozenset(["arm1", "arm2"])
