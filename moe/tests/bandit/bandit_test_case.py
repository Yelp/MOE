# -*- coding: utf-8 -*-
"""Base test case class for bandit tests; includes different historical infos (different sampled arms)."""
import testify as T

from moe.bandit.data_containers import HistoricalData, SampleArm


class BanditTestCase(T.TestCase):

    """Base test case for the bandit library.

    This sets up arms for test cases and includes an integration test case for
    verifying that default values do not throw an error.

    """

    bandit_class = None  # Define in a subclass

    """Set up arms for test cases."""
    one_arm_test_case = HistoricalData(sample_arms={"arm1": SampleArm(win=0, loss=0, total=0)})
    two_new_arms_test_case = HistoricalData(sample_arms={"arm1": SampleArm(win=0, loss=0, total=0), "arm2": SampleArm(win=0, loss=0, total=0)})
    two_arms_test_case = HistoricalData(sample_arms={"arm1": SampleArm(win=1, loss=0, total=1), "arm2": SampleArm(win=0, loss=0, total=0)})
    three_arms_test_case = HistoricalData(sample_arms={"arm1": SampleArm(win=2, loss=1, total=3), "arm2": SampleArm(win=1, loss=1, total=2), "arm3": SampleArm(win=0, loss=0, total=0)})
    three_arms_float_payoffs_test_case = HistoricalData(sample_arms={"arm1": SampleArm(win=2.2, loss=1.1, total=3), "arm2": SampleArm(win=2.1, loss=1.1, total=3), "arm3": SampleArm(win=0, loss=0, total=0)})

    historical_infos_to_test = [
                            one_arm_test_case,
                            two_new_arms_test_case,
                            two_arms_test_case,
                            three_arms_test_case,
                            three_arms_float_payoffs_test_case
                            ]

    def _test_init_default(self):
        """Verify that default values do not throw and error. This is purely an integration test."""
        for historical_info in self.historical_infos_to_test:
            bandit = self.bandit_class(historical_info=historical_info)
            bandit.allocate_arms()
            bandit.choose_arm()
