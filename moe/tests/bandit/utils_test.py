# -*- coding: utf-8 -*-
"""Tests for functions in utils."""
import logging

import testify as T

from moe.bandit.utils import get_equal_arm_allocations
from moe.tests.bandit.bandit_test_case import BanditTestCase


class UtilsTest(BanditTestCase):

    """Tests :func:`moe.bandit.utils.get_equal_arm_allocations`."""

    @T.class_setup
    def disable_logging(self):
        """Disable logging (for the duration of this test case)."""
        logging.disable(logging.CRITICAL)

    @T.class_teardown
    def enable_logging(self):
        """Re-enable logging (so other test cases are unaffected)."""
        logging.disable(logging.NOTSET)

    def test_empty_arm_invalid(self):
        """Test empty ``arms_sampled`` causes an ValueError."""
        T.assert_raises(ValueError, get_equal_arm_allocations, {})

    def test_get_equal_arm_allocations_no_winner(self):
        """Test allocations split among all sample arms when there is no winner."""
        T.assert_dicts_equal(
                get_equal_arm_allocations(self.two_unsampled_arms_test_case.arms_sampled),
                {"arm1": 0.5, "arm2": 0.5}
                )

    def test_get_equal_arm_allocations_one_winner(self):
        """Test all allocation given to the winning arm."""
        T.assert_dicts_equal(
                get_equal_arm_allocations(self.three_arms_test_case.arms_sampled, frozenset(["arm1"])),
                {"arm1": 1.0, "arm2": 0.0, "arm3": 0.0}
                )

    def test_get_equal_arm_allocations_two_winners(self):
        """Test allocations split between two winning arms."""
        T.assert_dicts_equal(
                get_equal_arm_allocations(self.three_arms_two_winners_test_case.arms_sampled, frozenset(["arm1", "arm2"])),
                {"arm1": 0.5, "arm2": 0.5, "arm3": 0.0}
                )


if __name__ == "__main__":
    T.run()
