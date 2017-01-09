# -*- coding: utf-8 -*-
"""Test class for bandit_ucb view."""
from moe.bandit.constant import BANDIT_UCB_ENDPOINT
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest.bandit_test import TestBanditViews
from moe.views.constant import BANDIT_UCB_MOE_ROUTE
from moe.views.rest.bandit_ucb import BanditUCBView


class TestBanditUCBViews(TestBanditViews):

    """Integration test for the /bandit/ucb endpoint."""

    _endpoint = BANDIT_UCB_ENDPOINT
    _historical_infos = BanditTestCase.historical_infos_to_test
    _moe_route = BANDIT_UCB_MOE_ROUTE
    _view = BanditUCBView

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        self._test_historical_info_passed_through()

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/ucb endpoint."""
        self._test_interface_returns_as_expected()
