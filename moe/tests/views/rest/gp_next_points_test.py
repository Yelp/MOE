# -*- coding: utf-8 -*-
"""Test class for gp_next_points_epi view."""
import simplejson as json

import testify as T

from moe.tests.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.views.constant import ALL_NEXT_POINTS_MOE_ROUTES, GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsResponse
from moe.optimal_learning.python.constant import TEST_OPTIMIZATION_MULTISTARTS, TEST_GRADIENT_DESCENT_PARAMETERS, TEST_OPTIMIZATION_NUM_RANDOM_SAMPLES
from moe.views.utils import _build_domain_info, _build_covariance_info


class TestGpNextPointsViews(RestGaussianProcessTestCase):

    """Test that the /gp/next_points/* endpoints do the same thing as the C++ interface."""

    precompute_gaussian_process_data = True

    def _build_json_payload(self, domain, gaussian_process, covariance, num_to_sample, lie_value=None):
        """Create a json_payload to POST to the /gp/next_points/* endpoint with all needed info."""
        dict_to_dump = {
            'num_to_sample': num_to_sample,
            'gp_info': self._build_gp_info(gaussian_process),
            'covariance_info': _build_covariance_info(covariance),
            'domain_info': _build_domain_info(domain),
            'optimization_info': {
                'num_multistarts': TEST_OPTIMIZATION_MULTISTARTS,
                'num_random_samples': TEST_OPTIMIZATION_NUM_RANDOM_SAMPLES,
                'optimization_parameters': dict(TEST_GRADIENT_DESCENT_PARAMETERS._asdict()),
                },
            }
        if lie_value is not None:
            dict_to_dump['lie_value'] = lie_value
        return json.dumps(dict_to_dump)

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/next_points/* endpoints do the same thing as the C++ interface."""
        for moe_route in ALL_NEXT_POINTS_MOE_ROUTES:
            for test_case in self.gp_test_environments:
                for num_to_sample in [1, 2, 4]:
                    python_domain, python_cov, python_gp = test_case

                    # Next point from REST
                    if moe_route.route_name == GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME:
                        json_payload = self._build_json_payload(python_domain, python_gp, python_cov, num_to_sample, lie_value=0.0)
                    else:
                        json_payload = self._build_json_payload(python_domain, python_gp, python_cov, num_to_sample)
                    resp = self.testapp.post(moe_route.endpoint, json_payload)
                    resp_schema = GpNextPointsResponse()
                    resp_dict = resp_schema.deserialize(json.loads(resp.body))
                    T.assert_in('points_to_sample', resp_dict)
                    T.assert_in('expected_improvement', resp_dict)

                    for ei in resp_dict['expected_improvement']:
                        T.assert_gte(ei, 0.0)

if __name__ == "__main__":
    T.run()
