# -*- coding: utf-8 -*-
"""Test class for bandit_ucb view."""
import simplejson as json

import testify as T

from moe.bandit.constant import BANDIT_UCB_ENDPOINT
from moe.tests.views.rest.bandit_test import TestBanditViews
from moe.views.constant import BANDIT_UCB_MOE_ROUTE
from moe.views.rest.bandit_ucb import BanditUCBView


class TestBanditUCBViews(TestBanditViews):

    """Integration test for the /bandit/ucb endpoint."""

    _endpoint = BANDIT_UCB_ENDPOINT
    _moe_route = BANDIT_UCB_MOE_ROUTE
    _view = BanditUCBView

    def _build_json_payload(self, subtype, historical_info):
        """Create a json_payload to POST to the /bandit/ucb endpoint with all needed info."""
        dict_to_dump = {
            'subtype': subtype,
            'historical_info': historical_info.json_payload(),
            }

        return json.dumps(dict_to_dump)

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        self._test_historical_info_passed_through()

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/ucb endpoint."""
        self._test_interface_returns_as_expected()


if __name__ == "__main__":
    T.run()
