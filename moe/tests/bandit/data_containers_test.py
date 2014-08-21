# -*- coding: utf-8 -*-
"""Tests for functions in SampleArm and HistoricalData."""
import copy

import logging

import pprint

import testify as T

from moe.bandit.data_containers import HistoricalData, SampleArm
from moe.tests.bandit.bandit_test_case import BanditTestCase


class DataContainersTest(BanditTestCase):

    """Tests functions in :class:`moe.bandit.data_containers.SampleArm` and :class:`moe.bandit.data_containers.HistoricalData`."""

    @T.class_setup
    def disable_logging(self):
        """Disable logging (for the duration of this test case)."""
        logging.disable(logging.CRITICAL)

    @T.class_teardown
    def enable_logging(self):
        """Re-enable logging (so other test cases are unaffected)."""
        logging.disable(logging.NOTSET)

    def test_sample_arm_str(self):
        """Test SampleArm's __str__ overload operator."""
        for historical_info in self.historical_infos_to_test:
            for arm in historical_info.arms_sampled.itervalues():
                T.assert_equals(str(arm), pprint.pformat(arm.json_payload()))

    def test_sample_arm_add(self):
        """Test SampleArm's __add__ overload operator."""
        arm1 = SampleArm(win=2, loss=1, total=3)
        arm2 = SampleArm(win=3, loss=2, total=5)
        arm3 = arm1 + arm2
        T.assert_equals(arm3.json_payload(), SampleArm(win=5, loss=3, total=8).json_payload())
        # Verify that the + operator does not modify arm1 and arm2
        T.assert_equals(arm1.json_payload(), SampleArm(win=2, loss=1, total=3).json_payload())
        T.assert_equals(arm2.json_payload(), SampleArm(win=3, loss=2, total=5).json_payload())

        arm1 += arm2
        arm2 += arm1
        T.assert_equals(arm1.json_payload(), SampleArm(win=5, loss=3, total=8).json_payload())
        T.assert_equals(arm2.json_payload(), SampleArm(win=8, loss=5, total=13).json_payload())
        # Verify that modifying arm1 and arm2 does not change arm3
        T.assert_equals(arm3.json_payload(), SampleArm(win=5, loss=3, total=8).json_payload())

    def test_sample_arm_iadd(self):
        """Test SampleArm's __iadd__ overload operator. Verify that after x += y, x still retains its old id."""
        arm1 = SampleArm(win=2, loss=1, total=3)
        arm1_old_id = id(arm1)
        arm1 += SampleArm(win=3, loss=2, total=5)
        arm1_new_id = id(arm1)
        T.assert_equals(arm1_old_id, arm1_new_id)

    def test_sample_arm_add_arm_with_variance_invalid(self):
        """Test that adding arms with variance causes a ValueError. Neither of the arms can have non-None variance."""
        arm = SampleArm(win=2, loss=1, total=500, variance=0.1)
        T.assert_raises(ValueError, arm.__add__, SampleArm(win=2, loss=1, total=500, variance=None))
        arm = SampleArm(win=2, loss=1, total=500, variance=None)
        T.assert_raises(ValueError, arm.__add__, SampleArm(win=2, loss=1, total=500, variance=0.1))

    def test_historical_data_str(self):
        """Test HistoricalData's __str__ overload operator."""
        for historical_info in self.historical_infos_to_test:
            T.assert_equals(str(historical_info), pprint.pformat(historical_info.json_payload()))

    def test_historical_data_append_unsampled_arm(self):
        """Test that adding an unsampled arm (already exists in historical info) to HistoricalData does not change anything."""
        historical_info = self.two_unsampled_arms_test_case
        historical_info.append_sample_arms(self.one_arm_test_case.arms_sampled)
        T.assert_dicts_equal(
                historical_info.json_payload(),
                self.two_unsampled_arms_test_case.json_payload()
                )

    def test_historical_data_append_arms(self):
        """Test that appending arms to HistoricalData updates historical info correctly."""
        historical_info = copy.deepcopy(self.three_arms_test_case)
        historical_info.append_sample_arms(self.three_arms_two_winners_test_case.arms_sampled)
        expected_historical_info = HistoricalData(
            sample_arms={
                "arm1": SampleArm(win=4, loss=2, total=6),
                "arm2": SampleArm(win=3, loss=2, total=5),
                "arm3": SampleArm(win=0, loss=0, total=0),
                }
            )
        T.assert_dicts_equal(
                historical_info.json_payload(),
                expected_historical_info.json_payload()
                )

    def test_historical_data_append_arms_with_variance_invalid(self):
        """Test that adding arms with variance causes a ValueError."""
        historical_info = copy.deepcopy(self.three_arms_with_variance_no_unsampled_arm_test_case)
        T.assert_raises(ValueError, historical_info.append_sample_arms, self.three_arms_with_variance_no_unsampled_arm_test_case.arms_sampled)


if __name__ == "__main__":
    T.run()
