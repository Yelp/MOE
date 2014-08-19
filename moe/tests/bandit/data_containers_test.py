# -*- coding: utf-8 -*-
"""Tests for functions in SampleArm and HistoricalData."""
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
        historical_info = self.three_arms_test_case
        historical_info.append_sample_arms(self.three_arms_two_winners_test_case.arms_sampled)
        expected_historical_info = HistoricalData(sample_arms={"arm1": SampleArm(win=4, loss=2, total=6), "arm2": SampleArm(win=3, loss=2, total=5), "arm3": SampleArm(win=0, loss=0, total=0)})
        T.assert_dicts_equal(
                historical_info.json_payload(),
                expected_historical_info.json_payload()
                )

    def test_historical_data_append_arms_with_variance_invalid(self):
        """Test that adding arms with variance causes a ValueError."""
        historical_info = self.three_arms_with_variance_no_unsampled_arm_test_case
        T.assert_raises(ValueError, historical_info.append_sample_arms, self.three_arms_with_variance_no_unsampled_arm_test_case.arms_sampled)


if __name__ == "__main__":
    T.run()
