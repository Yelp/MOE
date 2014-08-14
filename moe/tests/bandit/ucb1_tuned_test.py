# -*- coding: utf-8 -*-
"""Test UCB1-tuned bandit implementation.

Test default values with one, two, and three arms.
Test different cases including unsampled arms and multiple winners.

"""
import testify as T

from moe.bandit.ucb1_tuned import UCB1Tuned
from moe.tests.bandit.ucb_test_case import UCBTestCase


class UCB1TunedTest(UCBTestCase):

    """Verify that different historical infos return correct results."""

    bandit_class = UCB1Tuned

    def test_init_default(self):
        """Verify that default values do not throw and error. This is purely an integration test."""
        self._test_init_default()

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        bandit = self.bandit_class(self.one_arm_test_case)
        self._test_one_arm(bandit)

    def test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always allocate each arm equally (the allocation is 0.5 for both arms). This tests num_unsampled_arms == num_arms > 1."""
        self._test_two_unsampled_arms()

    def test_three_arms_one_unsampled_arm(self):
        """Check that the three-arms cases with integer and float payoffs return the expected arm allocations. When arm3 is the only unsampled arm, we expect all allocation is given to arm3."""
        self._test_three_arms_one_unsampled_arm()

    def test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return the expected arm allocations. This tests num_arms > num_winning_arms > 1."""
        self._test_three_arms_two_winners()

    def test_three_arms_diferent_variance(self):
        """Check that the three-arms cases with different variance (same average payoff) return the expected arm allocations. The highest variance wins."""
        bandit = self.bandit_class(self.three_arms_with_variance_no_unsampled_arm_test_case)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 1.0, "arm2": 0.0, "arm3": 0.0})


if __name__ == "__main__":
    T.run()
