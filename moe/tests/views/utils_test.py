# -*- coding: utf-8 -*-
"""Tests for functions in utils."""
from moe.bandit.data_containers import BernoulliArm
from moe.views.utils import _make_bandit_historical_info_from_params
from moe.tests.bandit.bandit_test_case import BanditTestCase


class TestUtils(BanditTestCase):

    """Tests :func:`moe.views.utils._make_bandit_historical_info_from_params`."""

    def make_params_from_bandit_historical_info(self, historical_info):
        """Create params from given ``historical_info``."""
        return {
            'historical_info': historical_info.json_payload(),
            }

    def test_make_bandit_historical_info_from_params_make_bernoulli_arms(self):
        """Test that the function can make historical infos with Bernoulli arms."""
        historical_info = self.three_arms_with_variance_no_unsampled_arm_test_case
        for historical_info in self.bernoulli_historical_infos_to_test:
            assert _make_bandit_historical_info_from_params(self.make_params_from_bandit_historical_info(historical_info), BernoulliArm).json_payload() == historical_info.json_payload()

    def test_make_bandit_historical_info_from_params_variance_passed_through(self):
        """Test that the variance of a given sample arm got passed through."""
        historical_info = self.three_arms_with_variance_no_unsampled_arm_test_case
        assert _make_bandit_historical_info_from_params(self.make_params_from_bandit_historical_info(historical_info)).json_payload() == historical_info.json_payload()
