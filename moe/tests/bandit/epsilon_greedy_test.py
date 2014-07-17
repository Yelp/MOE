# -*- coding: utf-8 -*-
"""Test epsilon-greedy bandit implementation.

Test default values with one, two, and three arms.
Test one arm with various epsilon values.

"""
import testify as T

from moe.bandit.epsilon_greedy import EpsilonGreedy
from moe.tests.bandit.epsilon_test_case import EpsilonTestCase


class EpsilonGreedyTest(EpsilonTestCase):

    """Verify that different epsilon values and historical infos return correct results."""

    bandit_class = EpsilonGreedy

    def test_init_default(self):
        """Verify that default values do not throw and error."""
        self._test_init_default()

    def test_one_arm(self):
        """Check that the one-arm case always returns the given arm as the winning arm and the allocation is 1.0."""
        for epsilon in self.epsilons_to_test:
            bandit = self.bandit_class(self.one_arm, epsilon)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 1.0})
            T.assert_equal(bandit.choose_arm(), "arm1")

    def test_two_new_arms(self):
        """Check that the two-new-arms case always allocate each arm equally (the allocation is 0.5 for both arms)."""
        for epsilon in self.epsilons_to_test:
            bandit = self.bandit_class(self.two_new_arms, epsilon)
            T.assert_dicts_equal(bandit.allocate_arms(), {"arm1": 0.5, "arm2": 0.5})
 
if __name__ == "__main__":
    T.run()
