# -*- coding: utf-8 -*-
"""Test class for gp_mean_var view."""
import simplejson as json
import testify as T

from moe.optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points
from moe.tests.moe.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.views.gp_mean_var import GpMeanVarResponse
from moe.views.constant import GP_MEAN_VAR_ENDPOINT


class TestGpMeanVarView(RestGaussianProcessTestCase):

    """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""

    endpoint = GP_MEAN_VAR_ENDPOINT

    test_cases = [
            {
                'domain': RestGaussianProcessTestCase.domain_1d,
                'points_to_sample': get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_1d),
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_2d,
                'points_to_sample': get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_2d),
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_3d,
                'points_to_sample': get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_3d),
                'num_points_in_sample': 10,
                },
            ]

    def _build_json_payload(self, GP, points_to_sample):
        """Create a json_payload to POST to the /gp/mean_var endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_sample': points_to_sample,
            'gp_info': self._build_gp_info(GP),
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""
        for test_case in self.test_cases:
            points_to_sample = test_case['points_to_sample']
            num_points_in_sample = test_case['num_points_in_sample']
            domain = test_case['domain']

            GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)
            # EI from C++
            cpp_mean, cpp_var = GP.get_mean_and_var_of_points(points_to_sample)

            # EI from REST
            json_payload = self._build_json_payload(GP, points_to_sample.tolist())
            resp = self.testapp.post(self.endpoint, json_payload)
            resp_schema = GpMeanVarResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_mean = resp_dict.get('mean')
            rest_var = resp_dict.get('var')

            self.assert_lists_relatively_equal(rest_mean, cpp_mean, tol=1e-11)
            self.assert_matrix_relatively_equal(rest_var, cpp_var, tol=1e-11)

if __name__ == "__main__":
    T.run()
