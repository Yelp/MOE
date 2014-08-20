# -*- coding: utf-8 -*-
"""Test class for bandit_epsilon view."""
import pyramid.testing

import simplejson as json

import testify as T

from moe.bandit.constant import BANDIT_EPSILON_ENDPOINT, EPSILON_SUBTYPES_TO_DEFAULT_HYPERPARAMETER_INFOS, EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY
from moe.tests.bandit.bandit_test_case import BanditTestCase
from moe.tests.views.rest.bandit_test import TestBanditViews
from moe.views.constant import BANDIT_EPSILON_MOE_ROUTE
from moe.views.rest.bandit_epsilon import BanditEpsilonView


class TestBanditEpsilonViews(TestBanditViews):

    """Integration test for the /bandit/epsilon endpoint."""

    _endpoint = BANDIT_EPSILON_ENDPOINT
    _historical_infos = BanditTestCase.historical_infos_to_test
    _moe_route = BANDIT_EPSILON_MOE_ROUTE
    _view = BanditEpsilonView

    def _build_json_payload(self, subtype, historical_info, hyperparameter_info=None):
        """Create a json_payload to POST to the /bandit/epsilon endpoint with all needed info."""
        if hyperparameter_info is None:
            hyperparameter_info = EPSILON_SUBTYPES_TO_DEFAULT_HYPERPARAMETER_INFOS[subtype]
        dict_to_dump = {
            'subtype': subtype,
            'historical_info': historical_info.json_payload(),
            'hyperparameter_info': hyperparameter_info,
            }

        return json.dumps(dict_to_dump)

    def test_epsilon_greedy_hyperparameters_passed_through(self):
        """Test that the hyperparameters get passed through to the epsilon-greedy endpoint."""
        historical_info = self.one_arm_test_case

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(EPSILON_SUBTYPE_GREEDY, historical_info, EPSILON_SUBTYPES_TO_DEFAULT_HYPERPARAMETER_INFOS[EPSILON_SUBTYPE_GREEDY]))

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

    def test_epsilon_first_hyperparameters_passed_through(self):
        """Test that the hyperparameters get passed through to the epsilon-first endpoint."""
        historical_info = self.one_arm_test_case

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(EPSILON_SUBTYPE_FIRST, historical_info, EPSILON_SUBTYPES_TO_DEFAULT_HYPERPARAMETER_INFOS[EPSILON_SUBTYPE_FIRST]))

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = self._view(request)
        params = view.get_params_from_request()

        T.assert_dicts_equal(params['hyperparameter_info'], json_payload['hyperparameter_info'])

        # Test an arbitrary epsilon and total_tamples get passed through
        json_payload['hyperparameter_info']['epsilon'] = 1.0
        json_payload['hyperparameter_info']['total_samples'] = 20000

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = self._view(request)
        params = view.get_params_from_request()

        T.assert_dicts_equal(params['hyperparameter_info'], json_payload['hyperparameter_info'])

    def test_historical_info_passed_through(self):
        """Test that the historical info get passed through to the endpoint."""
        self._test_historical_info_passed_through()

    def test_interface_returns_as_expected(self):
        """Integration test for the /bandit/epsilon endpoint."""
        self._test_interface_returns_as_expected()


if __name__ == "__main__":
    T.run()
