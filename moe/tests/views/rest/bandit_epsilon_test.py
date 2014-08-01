# -*- coding: utf-8 -*-
"""Test class for bandit_epsilon view."""
import pyramid.testing

import simplejson as json

import testify as T

from moe.bandit.constant import DEFAULT_EPSILON, EPSILON_SUBTYPE_GREEDY
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import BANDIT_EPSILON_MOE_ROUTE
from moe.views.rest.bandit_epsilon import BanditEpsilonResponse, BanditEpsilonView


class TestBanditEpsilonViews(BanditTestCase, RestTestCase):

    """Integration test for the /bandit/epsilon endpoint."""

    def _build_json_payload(self, subtype, historical_info, epsilon):
        """Create a json_payload to POST to the /bandit/epsilon endpoint with all needed info."""
        dict_to_dump = {
            'subtype': subtype,
            'historical_info': historical_info.json_payload(),
            'hyperparameter_info': {
                'epsilon': epsilon,
                },
            }

        return json.dumps(dict_to_dump)

    def test_hyperparameters_passed_through(self):
        """Test that the hyperparameters get passed through to the endpoint."""
        historical_info = self.one_arm_test_case

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(EPSILON_SUBTYPE_GREEDY, historical_info, DEFAULT_EPSILON))

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = BanditEpsilonView(request)
        params = view.get_params_from_request()

        T.assert_dicts_equal(params['hyperparameter_info'], json_payload['hyperparameter_info'])

        # Test arbitrary epsilons get passed through
        json_payload['hyperparameter_info']['epsilon'] = 1.0

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = BanditEpsilonView(request)
        params = view.get_params_from_request()

        T.assert_dicts_equal(params['hyperparameter_info'], json_payload['hyperparameter_info'])

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        for historical_info in self.historical_infos_to_test:
            # Test default test parameters get passed through
            json_payload = json.loads(self._build_json_payload(EPSILON_SUBTYPE_GREEDY, historical_info, DEFAULT_EPSILON))

            request = pyramid.testing.DummyRequest(post=json_payload)
            request.json_body = json_payload
            view = BanditEpsilonView(request)
            params = view.get_params_from_request()

            T.assert_dicts_equal(params['historical_info'], json_payload['historical_info'])

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/epsilon endpoint."""
        moe_route = BANDIT_EPSILON_MOE_ROUTE
        for historical_info in self.historical_infos_to_test:
            json_payload = self._build_json_payload(EPSILON_SUBTYPE_GREEDY, historical_info, DEFAULT_EPSILON)
            arm_names = set([arm_name for arm_name in historical_info.arms_sampled.iterkeys()])
            resp = self.testapp.post(moe_route.endpoint, json_payload)
            resp_schema = BanditEpsilonResponse()
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
