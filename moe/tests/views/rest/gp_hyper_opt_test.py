# -*- coding: utf-8 -*-
"""Test class for gp_hyper_opt view."""
import pyramid.testing

import simplejson as json

from moe.optimal_learning.python.constant import TEST_OPTIMIZER_MULTISTARTS, TEST_GRADIENT_DESCENT_PARAMETERS, TEST_OPTIMIZER_NUM_RANDOM_SAMPLES, GRADIENT_DESCENT_OPTIMIZER
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import GP_HYPER_OPT_MOE_ROUTE
from moe.views.rest.gp_hyper_opt import GpHyperOptView
from moe.views.schemas.rest.gp_hyper_opt import GpHyperOptResponse
from moe.views.utils import _make_optimizer_parameters_from_params


class TestGpHyperOptViews(GaussianProcessTestCase, RestTestCase):

    """Integration test for the /gp/hyper_opt endpoint."""

    precompute_gaussian_process_data = True

    @staticmethod
    def _build_json_payload(domain, covariance, historical_data):
        """Create a json_payload to POST to the /gp/hyper_opt endpoint with all needed info."""
        hyper_dim = domain.dim + 1
        dict_to_dump = {
            'gp_historical_info': historical_data.json_payload(),
            'covariance_info': covariance.get_json_serializable_info(),
            'domain_info': domain.get_json_serializable_info(minimal=True),
            'optimizer_info': {
                'optimizer_type': GRADIENT_DESCENT_OPTIMIZER,
                'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                'optimizer_parameters': dict(TEST_GRADIENT_DESCENT_PARAMETERS._asdict()),
                },
            'hyperparameter_domain_info': {
                'dim': hyper_dim,
                'domain_type': 'tensor_product',
                'domain_bounds': [],
                },
            }

        for _ in range(hyper_dim):
            dict_to_dump['hyperparameter_domain_info']['domain_bounds'].append({
                'min': 0.1,
                'max': 2.0,
                })
        return json.dumps(dict_to_dump)

    def test_hyperparameters_passed_through(self):
        """Test that the hyperparameters get passed through to the endpoint."""
        test_case = self.gp_test_environments[0]

        python_domain, python_gp = test_case
        python_cov, historical_data = python_gp.get_core_data_copy()

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(python_domain, python_cov, historical_data))

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpHyperOptView(request)
        params = view.get_params_from_request()

        assert params['hyperparameter_domain_info'] == json_payload['hyperparameter_domain_info']

        # Test arbitrary parameters get passed through
        json_payload['hyperparameter_domain_info']['domain_bounds'] = []
        for i in range(json_payload['hyperparameter_domain_info']['dim']):
            json_payload['hyperparameter_domain_info']['domain_bounds'].append({
                'min': 0.2 * i,
                'max': 0.5 * i,
                })

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpHyperOptView(request)
        params = view.get_params_from_request()

        assert params['hyperparameter_domain_info'] == json_payload['hyperparameter_domain_info']

    def test_optimizer_params_passed_through(self):
        """Test that the optimizer parameters get passed through to the endpoint."""
        test_case = self.gp_test_environments[0]

        python_domain, python_gp = test_case
        python_cov, historical_data = python_gp.get_core_data_copy()

        # Test default test parameters get passed through
        json_payload = json.loads(self._build_json_payload(python_domain, python_cov, historical_data))

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpHyperOptView(request)
        params = view.get_params_from_request()
        _, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

        test_param_dict = TEST_GRADIENT_DESCENT_PARAMETERS._asdict()
        test_param_dict['num_multistarts'] = TEST_OPTIMIZER_MULTISTARTS
        assert optimizer_parameters._get_member_dict() == test_param_dict

        # Test arbitrary parameters get passed through
        for i, key in enumerate(test_param_dict.iterkeys()):
            test_param_dict[key] /= 2
        test_num_multistarts = test_param_dict.pop('num_multistarts')

        json_payload['optimizer_info']['num_multistarts'] = test_num_multistarts
        json_payload['optimizer_info']['optimizer_parameters'] = test_param_dict

        request = pyramid.testing.DummyRequest(post=json_payload)
        request.json_body = json_payload
        view = GpHyperOptView(request)
        params = view.get_params_from_request()
        _, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

        test_param_dict['num_multistarts'] = test_num_multistarts
        assert optimizer_parameters._get_member_dict() == test_param_dict

    def test_interface_returns_same_as_cpp(self):
        """Integration test for the /gp/hyper_opt endpoint."""
        moe_route = GP_HYPER_OPT_MOE_ROUTE
        for test_case in self.gp_test_environments:
            python_domain, python_gp = test_case
            python_cov, historical_data = python_gp.get_core_data_copy()

            json_payload = self._build_json_payload(python_domain, python_cov, historical_data)
            resp = self.testapp.post(moe_route.endpoint, json_payload)
            resp_schema = GpHyperOptResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))

            assert 'covariance_info' in resp_dict
            assert resp_dict['covariance_info']['covariance_type'] == python_cov.covariance_type
            # The optimal hyperparameters should be greater than zero
            for hyperparameter in resp_dict['covariance_info']['hyperparameters']:
                assert hyperparameter >= 0.0
