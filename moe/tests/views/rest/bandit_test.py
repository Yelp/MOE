# -*- coding: utf-8 -*-
"""Test class for bandit views."""
import pyramid.testing

import simplejson as json

from moe.bandit.linkers import BANDIT_ENDPOINTS_TO_SUBTYPES
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.schemas.bandit_pretty_view import BanditResponse


class TestBanditViews(BanditTestCase, RestTestCase):

    """Integration test for the /bandit endpoints."""

    _endpoint = None  # Define in a subclass
    _historical_infos = None  # Define in a subclass
    _moe_route = None  # Define in a subclass
    _view = None  # Define in a subclass

    @staticmethod
    def _build_json_payload(subtype, historical_info):
        """Create a json_payload to POST to the /bandit/* endpoint with all needed info."""
        dict_to_dump = {
            'subtype': subtype,
            'historical_info': historical_info.json_payload(),
            }

        return json.dumps(dict_to_dump)

    def _test_historical_info_passed_through(self):
        """Test that the historical infos get passed through to the endpoint."""
        for subtype in BANDIT_ENDPOINTS_TO_SUBTYPES[self._endpoint]:
            for historical_info in self._historical_infos:
                # Test default test parameters get passed through
                json_payload = json.loads(self._build_json_payload(subtype, historical_info))

                request = pyramid.testing.DummyRequest(post=json_payload)
                request.json_body = json_payload
                view = self._view(request)
                params = view.get_params_from_request()

                assert params['historical_info'] == json_payload['historical_info']

    def _test_interface_returns_as_expected(self):
        """Integration test for the bandit endpoints."""
        for subtype in BANDIT_ENDPOINTS_TO_SUBTYPES[self._endpoint]:
            for historical_info in self._historical_infos:
                json_payload = self._build_json_payload(subtype, historical_info)
                arm_names = set([arm_name for arm_name in historical_info.arms_sampled.iterkeys()])
                resp = self.testapp.post(self._moe_route.endpoint, json_payload)
                resp_schema = BanditResponse()
                resp_dict = resp_schema.deserialize(json.loads(resp.body))
                resp_arm_names = set([arm_name for arm_name in resp_dict['arm_allocations'].iterkeys()])
                assert arm_names == resp_arm_names
                # The allocations should be in range [0, 1]
                # The sum of all allocations should be 1.0.
                total_allocation = 0
                for allocation in resp_dict['arm_allocations'].itervalues():
                    assert allocation >= 0
                    assert allocation <= 1
                    total_allocation += allocation
                assert total_allocation == 1.0
