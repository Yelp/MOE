# -*- coding: utf-8 -*-
"""Test UCB1 bandit implementation.

Test that the one arm case returns the given arm as the winning arm.
Test that two-arm cases with specified random seeds return expected results.

"""
import numpy

from moe.bandit.bla.bla import BLA
from moe.bandit.constant import DEFAULT_BLA_SUBTYPE
from moe.tests.bandit.bandit_test_case import BanditTestCase


class TestBLA(BanditTestCase):

    """Verify that different historical infos return correct results."""

    bandit_class = BLA

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        bandit = self.bandit_class(self.one_arm_test_case)
        self._test_one_arm(bandit)

    def test_two_arms_one_winner(self):
        """Check that the two-arms case with random seed 0 always allocate arm1:1.0 and arm2:0.0."""
        old_state = numpy.random.get_state()
        numpy.random.seed(0)
        bandit = self.bandit_class(self.two_arms_test_case, DEFAULT_BLA_SUBTYPE)
        arms_to_allocations = bandit.allocate_arms()
        assert arms_to_allocations == {"arm1": 1.0, "arm2": 0.0}
        assert bandit.choose_arm(arms_to_allocations) == "arm1"
        numpy.random.set_state(old_state)
