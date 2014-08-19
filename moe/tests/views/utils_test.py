# -*- coding: utf-8 -*-
"""Tests for functions in utils."""
import testify as T

from moe.views.utils import _make_bandit_historical_info_from_params
from moe.tests.bandit.bandit_test_case import BanditTestCase


class UtilsTest(BanditTestCase):

    """Tests :func:`moe.bandit.views..utils._make_bandit_historical_info_from_params`."""

    def make_params_from_bandit_historical_info(self, historical_info):
        """Create params from given ``historical_info``."""
        return {
            'historical_info': historical_info.json_payload(),
            }

    def test_make_bandit_historical_info_from_params_variance_passed_through(self):
        """Test that the variance of a given sample arm got passed through."""
        historical_info = self.three_arms_with_variance_no_unsampled_arm_test_case
        T.assert_equals(
                _make_bandit_historical_info_from_params(self.make_params_from_bandit_historical_info(historical_info)),
                historical_info
                )


if __name__ == "__main__":
    T.run()
