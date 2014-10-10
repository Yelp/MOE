# -*- coding: utf-8 -*-
"""Test epsilon-greedy bandit implementation.

Test default values with one, two, and three arms.
Test one arm with various epsilon values.

"""
from moe.bandit.epsilon.epsilon_greedy import EpsilonGreedy
from moe.tests.bandit.epsilon_test_case import EpsilonTestCase


class TestEpsilonGreedy(EpsilonTestCase):

    """Verify that different epsilon values and historical infos return correct results."""

    bandit_class = EpsilonGreedy

    def test_init_default(self):
        """Verify that default values do not throw and error. This is purely an integration test."""
        self._test_init_default()

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        for epsilon in self.epsilons_to_test:
            bandit = self.bandit_class(self.one_arm_test_case, epsilon)
            self._test_one_arm(bandit)

    def test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always allocate each arm equally (the allocation is 0.5 for both arms). This tests num_winning_arms == num_arms > 1."""
        for epsilon in self.epsilons_to_test:
            bandit = self.bandit_class(self.two_unsampled_arms_test_case, epsilon)
            assert bandit.allocate_arms() == {"arm1": 0.5, "arm2": 0.5}

    def test_two_arms_epsilon_zero(self):
        """Check that the two-arms case with zero epsilon always allocate arm1:1.0 and arm2:0.0 when average payoffs are arm1:1.0 and arm2:0.0."""
        epsilon = 0.0
        bandit = self.bandit_class(self.two_arms_test_case, epsilon)
        arms_to_allocations = bandit.allocate_arms()
        assert arms_to_allocations == {"arm1": 1.0, "arm2": 0.0}
        assert bandit.choose_arm(arms_to_allocations) == "arm1"

    def test_two_arms_epsilon_one(self):
        """Check that the two-arms case with one epsilon always allocate arm1:0.5 and arm2:0.5 when average payoffs are arm1:1.0 and arm2:0.0."""
        epsilon = 1.0
        bandit = self.bandit_class(self.two_arms_test_case, epsilon)
        assert bandit.allocate_arms() == {"arm1": 0.5, "arm2": 0.5}

    def test_three_arms(self):
        """Check that the three-arms cases with integer and float payoffs return the expected arm allocations."""
        epsilon = 0.03
        for historical_info in [self.three_arms_test_case, self.three_arms_float_payoffs_test_case]:
            bandit = self.bandit_class(historical_info, epsilon)
            assert bandit.allocate_arms() == {"arm1": 0.98, "arm2": 0.01, "arm3": 0.01}

    def test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return the expected arm allocations. This tests num_arms > num_winning_arms > 1."""
        epsilon = 0.03
        bandit = self.bandit_class(self.three_arms_two_winners_test_case, epsilon)
        assert bandit.allocate_arms() == {"arm1": 0.495, "arm2": 0.495, "arm3": 0.01}
