# -*- coding: utf-8 -*-
"""Test epsilon-first bandit implementation.

Test default values with one, two, and three arms.
Test one arm with various epsilon values.

"""
import testify as T

from moe.bandit.epsilon_first import EpsilonFirst
from moe.tests.bandit.epsilon_test_case import EpsilonTestCase


class EpsilonFirstTest(EpsilonTestCase):

    """Verify that different epsilon values and historical infos return correct results."""

    bandit_class = EpsilonFirst

    total_samples_to_test = [1, 10, 100]

    def test_init_default(self):
        """Verify that default values do not throw and error. This is purely an integration test."""
        self._test_init_default()

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        for epsilon in self.epsilons_to_test:
            for total_samples in self.total_samples_to_test:
                bandit = self.bandit_class(self.one_arm_test_case, epsilon, total_samples)
                T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 1.0})
                T.assert_equal(bandit.choose_arm(), "arm1")

    def test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always allocate each arm equally (the allocation is 0.5 for both arms). This tests num_winning_arms == num_arms > 1."""
        for epsilon in self.epsilons_to_test:
            for total_samples in self.total_samples_to_test:
                bandit = self.bandit_class(self.two_unsampled_arms_test_case, epsilon, total_samples)
                T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5})

    def test_two_arms_epsilon_zero(self):
        """Check that the two-arms case with zero epsilon (always exploit) always allocate arm1:1.0 and arm2:0.0 when average payoffs are arm1:1.0 and arm2:0.0."""
        epsilon = 0.0
        bandit = self.bandit_class(self.two_arms_test_case, epsilon)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 1.0, "arm2": 0.0})
        T.assert_equal(bandit.choose_arm(), "arm1")

    def test_two_arms_epsilon_one(self):
        """Check that the two-arms case with one epsilon (always explore) always allocate arm1:0.5 and arm2:0.5 when average payoffs are arm1:1.0 and arm2:0.0."""
        epsilon = 1.0
        bandit = self.bandit_class(self.two_arms_test_case, epsilon)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5})

    def test_three_arms_explore(self):
        """Check that the three-arms cases with integer and float payoffs in exploration phase return the expected arm allocations."""
        epsilon = 0.7
        total_samples = 10
        equal_allocation = 1.0 / 3
        for historical_info in [self.three_arms_test_case, self.three_arms_float_payoffs_test_case]:
            bandit = self.bandit_class(historical_info, epsilon, total_samples)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": equal_allocation, "arm2": equal_allocation, "arm3": equal_allocation})

    def test_three_arms_exploit(self):
        """Check that the three-arms cases with integer and float payoffs in exploitation phase return the expected arm allocations."""
        epsilon = 0.5
        total_samples = 10
        for historical_info in [self.three_arms_test_case, self.three_arms_float_payoffs_test_case]:
            bandit = self.bandit_class(historical_info, epsilon, total_samples)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 1.0, "arm2": 0.0, "arm3": 0.0})

    def test_three_arms_exploit_two_winners(self):
        """Check that the three-arms cases with two winners in exploitation phase return the expected arm allocations. This tests num_arms > num_winning_arms > 1."""
        epsilon = 0.5
        total_samples = 10
        bandit = self.bandit_class(self.three_arms_two_winners_test_case, epsilon, total_samples)
        T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5, "arm3": 0.0})


if __name__ == "__main__":
    T.run()
