# -*- coding: utf-8 -*-
"""Base test case class for UCB tests; includes different cases where unsampled arms are winners."""
import testify as T

from moe.tests.bandit.bandit_test_case import BanditTestCase


class UCBTestCase(BanditTestCase):

    """Base test case for the UCB bandit library."""

    def _test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always allocate each arm equally (the allocation is 0.5 for both arms). This tests num_unsampled_arms == num_arms > 1."""
        bandit = self.bandit_class(self.two_unsampled_arms_test_case)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5})

    def _test_three_arms_one_unsampled_arm(self):
        """Check that the three-arms cases with integer and float payoffs return the expected arm allocations. When arm3 is the only unsampled arm, we expect all allocation is given to arm3."""
        for historical_info in [self.three_arms_test_case, self.three_arms_float_payoffs_test_case, self.three_arms_two_winners_test_case]:
            bandit = self.bandit_class(historical_info)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.0, "arm2": 0.0, "arm3": 1.0})

    def _test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return the expected arm allocations. This tests num_arms > num_winning_arms > 1."""
        bandit = self.bandit_class(self.three_arms_two_winners_no_unsampled_arm_test_case)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5, "arm3": 0.0})


if __name__ == "__main__":
    T.run()
