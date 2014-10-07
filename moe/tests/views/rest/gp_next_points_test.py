# -*- coding: utf-8 -*-
"""Test class for gp_next_points_epi view."""
import pyramid.testing

import simplejson as json

from moe.optimal_learning.python.constant import TEST_OPTIMIZER_MULTISTARTS, TEST_GRADIENT_DESCENT_PARAMETERS, TEST_LBFGSB_PARAMETERS, TEST_OPTIMIZER_NUM_RANDOM_SAMPLES, TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS, CONSTANT_LIAR_METHODS
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import ALL_NEXT_POINTS_MOE_ROUTES, GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_ENDPOINT, GP_NEXT_POINTS_EPI_ROUTE_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.schemas.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.utils import _make_optimizer_parameters_from_params


class TestGpNextPointsViews(GaussianProcessTestCase, RestTestCase):

    """Integration test for the /gp/next_points/* endpoints."""

    precompute_gaussian_process_data = True
    num_sampled_list = (1, 2, 10)

    def _build_json_payload(self, domain, covariance, historical_data, num_to_sample, lie_value=None, lie_method=None, l_bfgs_b=False):
        """Create a json_payload to POST to the /gp/next_points/* endpoint with all needed info."""
        if l_bfgs_b:
            dict_to_dump = {
                'num_to_sample': num_to_sample,
                'mc_iterations': TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
                'gp_historical_info': historical_data.json_payload(),
                'covariance_info': covariance.get_json_serializable_info(),
                'domain_info': domain.get_json_serializable_info(),
                'optimizer_info': {
                    'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                    'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                    'optimizer_parameters': dict(TEST_LBFGSB_PARAMETERS._asdict()),
                    },
                'mvndst_parameters': {
                    'releps': 1.0,
                    'maxpts_per_dim': 200,
                    },
                }
        else:
            dict_to_dump = {
                'num_to_sample': num_to_sample,
                'mc_iterations': TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
                'gp_historical_info': historical_data.json_payload(),
                'covariance_info': covariance.get_json_serializable_info(),
                'domain_info': domain.get_json_serializable_info(),
                'optimizer_info': {
                    'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                    'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                    'optimizer_parameters': dict(TEST_GRADIENT_DESCENT_PARAMETERS._asdict()),
                    },
                }

        if lie_value is not None:
            dict_to_dump['lie_value'] = lie_value
        if lie_method is not None:
            dict_to_dump['lie_method'] = lie_method

        return json.dumps(dict_to_dump)

    def test_optimizer_params_passed_through(self):
        """Test that the optimizer parameters get passed through to the endpoint."""
        # TODO(GH-305): turn this into a unit test by going through OptimizableGpPrettyView
        # and mocking out dependencies (instead of awkwardly constructing a more complex object).
        test_case = self.gp_test_environments[0]
        num_to_sample = 1

        python_domain, python_gp = test_case
        python_cov, historical_data = python_gp.get_core_data_copy()

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(python_domain, python_cov, historical_data, num_to_sample))

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpNextPointsPrettyView(request)
        # get_params_from_request() requires this field is set. value is arbitrary for now.
        # TODO(GH-305): mock out this and other members
        view._route_name = GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME
        params = view.get_params_from_request()
        _, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

        assert optimizer_parameters.num_multistarts == TEST_OPTIMIZER_MULTISTARTS
        assert optimizer_parameters._python_max_num_steps == TEST_GRADIENT_DESCENT_PARAMETERS.max_num_steps

        # Test arbitrary parameters get passed through
        json_payload['optimizer_info']['num_multistarts'] = TEST_OPTIMIZER_MULTISTARTS + 5
        json_payload['optimizer_info']['optimizer_parameters']['max_num_steps'] = TEST_GRADIENT_DESCENT_PARAMETERS.max_num_steps + 10

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpNextPointsPrettyView(request)
        # get_params_from_request() requires this field is set. value is arbitrary for now.
        # TODO(GH-305): mock out this and other members
        view._route_name = GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME
        params = view.get_params_from_request()
        _, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

        assert optimizer_parameters.num_multistarts == TEST_OPTIMIZER_MULTISTARTS + 5
        assert optimizer_parameters._python_max_num_steps == TEST_GRADIENT_DESCENT_PARAMETERS.max_num_steps + 10

    def test_all_constant_liar_methods_function(self):
        """Test that each contant liar ``lie_method`` runs to completion. This is an integration test."""
        for test_case in self.gp_test_environments:
            python_domain, python_gp = test_case
            python_cov, historical_data = python_gp.get_core_data_copy()

            for constant_liar_method in CONSTANT_LIAR_METHODS:

                json_payload = self._build_json_payload(
                        python_domain,
                        python_cov,
                        historical_data,
                        2,  # num_to_sample
                        lie_method=constant_liar_method,
                        )

                resp = self.testapp.post(GP_NEXT_POINTS_CONSTANT_LIAR_ENDPOINT, json_payload)
                resp_schema = GpNextPointsResponse()
                resp_dict = resp_schema.deserialize(json.loads(resp.body))

                assert 'points_to_sample' in resp_dict
                assert len(resp_dict['points_to_sample']) == 2  # num_to_sample
                assert len(resp_dict['points_to_sample'][0]) == python_gp.dim

                assert 'status' in resp_dict
                assert 'expected_improvement' in resp_dict['status']
                assert resp_dict['status']['expected_improvement'] >= 0.0

    def test_interface_returns_same_as_cpp(self):
        """Integration test for the /gp/next_points/* endpoints."""
        for moe_route in ALL_NEXT_POINTS_MOE_ROUTES:
            for test_case in self.gp_test_environments:
                for num_to_sample in (1, 2, 4):
                    python_domain, python_gp = test_case
                    python_cov, historical_data = python_gp.get_core_data_copy()

                    # Next point from REST
                    if moe_route.route_name == GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME:
                        json_payload = self._build_json_payload(python_domain, python_cov, historical_data, num_to_sample, lie_value=0.0)
                    elif moe_route.route_name == GP_NEXT_POINTS_EPI_ROUTE_NAME and num_to_sample > 1:
                        json_payload = self._build_json_payload(python_domain, python_cov, historical_data, num_to_sample, l_bfgs_b=True)
                    else:
                        json_payload = self._build_json_payload(python_domain, python_cov, historical_data, num_to_sample)
                    resp = self.testapp.post(moe_route.endpoint, json_payload)
                    resp_schema = GpNextPointsResponse()
                    resp_dict = resp_schema.deserialize(json.loads(resp.body))

                    assert 'points_to_sample' in resp_dict
                    assert len(resp_dict['points_to_sample']) == num_to_sample
                    assert len(resp_dict['points_to_sample'][0]) == python_gp.dim

                    assert 'status' in resp_dict
                    assert 'expected_improvement' in resp_dict['status']
                    assert resp_dict['status']['expected_improvement'] >= 0.0
