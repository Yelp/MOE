# -*- coding: utf-8 -*-
"""Test class for gp_mean_var view."""
import numpy

import simplejson as json

import testify as T

from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.tests.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.views.constant import GP_EI_ENDPOINT
from moe.views.rest.gp_ei import GpEiResponse
from moe.views.utils import _build_domain_info, _build_covariance_info


class TestGpEiView(RestGaussianProcessTestCase):

    """Test that the /gp/ei endpoint does the same thing as the C++ interface."""

    precompute_gaussian_process_data = True
    endpoint = GP_EI_ENDPOINT

    def _build_json_payload(self, domain, gaussian_process, covariance, points_to_evaluate):
        """Create a json_payload to POST to the /gp/ei endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_evaluate': points_to_evaluate,
            'points_being_sampled': [],
            'gp_historical_info': self._build_gp_historical_info(gaussian_process),
            'covariance_info': _build_covariance_info(covariance),
            'domain_info': _build_domain_info(domain),
            })

        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/ei endpoint does the same thing as the C++ interface."""
        tolerance = 1.0e-11
        for test_case in self.gp_test_environments:
            python_domain, python_cov, python_gp = test_case

            cpp_cov = SquareExponential(python_cov.hyperparameters)
            cpp_gp = GaussianProcess(cpp_cov, python_gp._historical_data)

            points_to_evaluate = python_domain.generate_uniform_random_points_in_domain(10)

            # EI from C++
            expected_improvement_evaluator = ExpectedImprovement(
                    cpp_gp,
                    None,
                    )
            # TODO(GH-99): Change test case to have the right shape:
            # (num_to_evaluate, num_to_sample, dim)
            # Here we assume the shape is (num_to_evaluate, dim) so we insert an axis, making num_to_sample = 1.
            # Also might be worth testing more num_to_sample values (will require manipulating C++ RNG state).
            cpp_expected_improvement = expected_improvement_evaluator.evaluate_at_point_list(
                    points_to_evaluate[:, numpy.newaxis, :],
                    )

            # EI from REST
            json_payload = self._build_json_payload(python_domain, python_gp, python_cov, points_to_evaluate.tolist())
            resp = self.testapp.post(self.endpoint, json_payload)
            resp_schema = GpEiResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_expected_improvement = numpy.asarray(resp_dict.get('expected_improvement'))

            self.assert_vector_within_relative(
                    rest_expected_improvement,
                    cpp_expected_improvement,
                    tolerance,
                    )


if __name__ == "__main__":
    T.run()
