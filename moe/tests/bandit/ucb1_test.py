# -*- coding: utf-8 -*-
"""Test UCB1 bandit implementation.

Test default values with one, two, and three arms.
Test different cases including unsampled arms and multiple winners.

"""
import testify as T

from moe.bandit.ucb1 import UCB1
from moe.tests.bandit.bandit_test_case import BanditTestCase


class UCB1Test(BanditTestCase):

    """Verify that different historical infos return correct results."""

    bandit_class = UCB1

    def test_init_default(self):
        """Verify that default values do not throw and error. This is purely an integration test."""
        self._test_init_default()

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        bandit = self.bandit_class(self.one_arm_test_case)
        self._test_one_arm(bandit)

    def test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always allocate each arm equally (the allocation is 0.5 for both arms). This tests num_unsampled_arms == num_arms > 1."""
        bandit = self.bandit_class(self.two_unsampled_arms_test_case)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5})

    def test_three_arms_one_unsampled_arm(self):
        """Check that the three-arms cases with integer and float payoffs return the expected arm allocations. When arm3 is the only unsampled arm, we expect all allocation is given to arm3."""
        for historical_info in [self.three_arms_test_case, self.three_arms_float_payoffs_test_case, self.three_arms_two_winners_test_case]:
            bandit = self.bandit_class(historical_info)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.0, "arm2": 0.0, "arm3": 1.0})

    def test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return the expected arm allocations. This tests num_arms > num_winning_arms > 1."""
        bandit = self.bandit_class(self.three_arms_two_winners_no_unsampled_arm_test_case)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5, "arm3": 0.0})


if __name__ == "__main__":
    T.run()
