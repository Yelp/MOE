# -*- coding: utf-8 -*-
"""Test class for bandit_bla view."""
from moe.bandit.constant import BANDIT_BLA_ENDPOINT
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest.bandit_test import TestBanditViews
from moe.views.constant import BANDIT_BLA_MOE_ROUTE
from moe.views.rest.bandit_bla import BanditBLAView


class TestBanditBLAViews(TestBanditViews):

    """Integration test for the /bandit/bla endpoint."""

    _endpoint = BANDIT_BLA_ENDPOINT
    _historical_infos = BanditTestCase.bernoulli_historical_infos_to_test
    _moe_route = BANDIT_BLA_MOE_ROUTE
    _view = BanditBLAView

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        self._test_historical_info_passed_through()

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/bla endpoint."""
        self._test_interface_returns_as_expected()
