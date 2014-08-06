# -*- coding: utf-8 -*-
"""Test class for bandit_ucb view."""
import pyramid.testing

import simplejson as json

import testify as T

from moe.bandit.constant import UCB_SUBTYPES
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import BANDIT_UCB_MOE_ROUTE
from moe.views.rest.bandit_ucb import BanditUCBView
from moe.views.schemas.bandit_pretty_view import BanditResponse


class TestBanditUCBViews(BanditTestCase, RestTestCase):

    """Integration test for the /bandit/ucb endpoint."""

    def _build_json_payload(self, subtype, historical_info):
        """Create a json_payload to POST to the /bandit/ucb endpoint with all needed info."""
        dict_to_dump = {
            'subtype': subtype,
            'historical_info': historical_info.json_payload(),
            }

        return json.dumps(dict_to_dump)

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        for subtype in UCB_SUBTYPES:
            for historical_info in self.historical_infos_to_test:
                # Test default test parameters get passed through
                json_payload = json.loads(self._build_json_payload(subtype, historical_info))

                request = pyramid.testing.DummyRequest(post=json_payload)
                request.json_body = json_payload
                view = BanditUCBView(request)
                params = view.get_params_from_request()

                T.assert_dicts_equal(params['historical_info'], json_payload['historical_info'])

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/ucb endpoint."""
        moe_route = BANDIT_UCB_MOE_ROUTE
        for subtype in UCB_SUBTYPES:
            for historical_info in self.historical_infos_to_test:
                json_payload = self._build_json_payload(subtype, historical_info)
                arm_names = set([arm_name for arm_name in historical_info.arms_sampled.iterkeys()])
                resp = self.testapp.post(moe_route.endpoint, json_payload)
                resp_schema = BanditResponse()
                resp_dict = resp_schema.deserialize(json.loads(resp.body))
                resp_arm_names = set([arm_name for arm_name in resp_dict['arm_allocations'].iterkeys()])
                T.assert_sets_equal(arm_names, resp_arm_names)
                # The allocations should be in range [0, 1]
                # The sum of all allocations should be 1.0.
                total_allocation = 0
                for allocation in resp_dict['arm_allocations'].itervalues():
                    T.assert_gte(allocation, 0)
                    T.assert_lte(allocation, 1)
                    total_allocation += allocation
                T.assert_equal(total_allocation, 1.0)


if __name__ == "__main__":
    T.run()
