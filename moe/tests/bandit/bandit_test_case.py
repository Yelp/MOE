# -*- coding: utf-8 -*-
"""Base test case class for bandit tests; includes different historical infos (different sampled arms)."""
import testify as T

from moe.bandit.data_containers import HistoricalData, SampleArm


class BanditTestCase(T.TestCase):

    """Base test case for the bandit library.

    This includes extra asserts for checking relative differences of floating point scalars/vectors and
    a routine to check that points are distinct.

    """

    bandit_class = None  # Define in a subclass

    """Set up arms for test cases."""
    one_arm = HistoricalData(sample_arms={"arm1": SampleArm(win=0, loss=0, total=0)})
    two_new_arms = HistoricalData(sample_arms={"arm1": SampleArm(win=0, loss=0, total=0), "arm2": SampleArm(win=0, loss=0, total=0)})
    two_arms = HistoricalData(sample_arms={"arm1": SampleArm(win=1, loss=0, total=1), "arm2": SampleArm(win=0, loss=0, total=0)})

    historical_infos_to_test = [one_arm, two_new_arms, two_arms]

    """Verify that default values do not throw and error.

    """
    def _test_init_default(self):
        for historical_info in self.historical_infos_to_test:
            bandit = self.bandit_class(historical_info=historical_info)
            bandit.allocate_arms()
            bandit.choose_arm()
