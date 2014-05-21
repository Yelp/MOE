# -*- coding: utf-8 -*-
"""Test class for gp_mean_var view."""
import simplejson as json

import testify as T

from moe.tests.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.views.constant import GP_MEAN_VAR_ENDPOINT
from moe.views.rest.gp_mean_var import GpMeanVarResponse
from moe.views.utils import _build_domain_info


class TestGpMeanVarView(RestGaussianProcessTestCase):

    """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""

    precompute_gaussian_process_data = True
    endpoint = GP_MEAN_VAR_ENDPOINT

    def _build_json_payload(self, domain, gaussian_process, covariance, points_to_sample):
        """Create a json_payload to POST to the /gp/mean_var endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_sample': points_to_sample,
            'gp_info': self._build_gp_info(gaussian_process),
            'covariance_info': self._build_covariance_info(covariance),
            'domain_info': _build_domain_info(domain),
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""
        for test_case in self.gp_test_environments:
            python_domain, python_cov, python_gp = test_case

            cpp_cov = SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = GaussianProcess(cpp_cov, python_gp._historical_data)

            points_to_sample = python_domain.generate_uniform_random_points_in_domain(10)

            # mean and var from C++
            cpp_mean = cpp_gp.compute_mean_of_points(points_to_sample)
            cpp_var = cpp_gp.compute_variance_of_points(points_to_sample)

            # mean and var from REST
            json_payload = self._build_json_payload(python_domain, python_gp, python_cov, points_to_sample.tolist())
            resp = self.testapp.post(self.endpoint, json_payload)
            resp_schema = GpMeanVarResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_mean = resp_dict.get('mean')
            rest_var = resp_dict.get('var')

            self.assert_lists_relatively_equal(rest_mean, cpp_mean, tol=1e-11)
            self.assert_matrix_relatively_equal(rest_var, cpp_var, tol=1e-11)

if __name__ == "__main__":
    T.run()
