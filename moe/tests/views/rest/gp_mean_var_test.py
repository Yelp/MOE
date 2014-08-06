# -*- coding: utf-8 -*-
"""Test class for gp_mean_var view."""
import numpy

import simplejson as json

import testify as T

from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import GP_MEAN_ENDPOINT, GP_VAR_ENDPOINT, GP_VAR_DIAG_ENDPOINT, GP_MEAN_VAR_ENDPOINT, GP_MEAN_VAR_DIAG_ENDPOINT
from moe.views.schemas.rest.gp_mean_var import GpMeanResponse, GpVarResponse, GpVarDiagResponse, GpMeanVarResponse, GpMeanVarDiagResponse


class TestGpMeanVarView(GaussianProcessTestCase, RestTestCase):

    """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""

    precompute_gaussian_process_data = True
    num_sampled_list = (1, 2, 3, 11, 20)
    endpoint = GP_MEAN_VAR_ENDPOINT

    def _build_json_payload(self, domain, covariance, historical_data, points_to_evaluate):
        """Create a json_payload to POST to the /gp/mean_var endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_evaluate': points_to_evaluate,
            'gp_historical_info': historical_data.json_payload(),
            'covariance_info': covariance.get_json_serializable_info(),
            'domain_info': domain.get_json_serializable_info(minimal=True),
            })
        return json_payload

    def test_mean_var_interface_returns_same_as_cpp(self):
        """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""
        tolerance = 1.0e-11
        for test_case in self.gp_test_environments:
            python_domain, python_gp = test_case
            python_cov, historical_data = python_gp.get_core_data_copy()

            cpp_cov = SquareExponential(python_cov.hyperparameters)
            cpp_gp = GaussianProcess(cpp_cov, historical_data)

            points_to_evaluate = python_domain.generate_uniform_random_points_in_domain(10)

            # mean and var from C++
            cpp_mean = cpp_gp.compute_mean_of_points(points_to_evaluate)
            cpp_var = cpp_gp.compute_variance_of_points(points_to_evaluate)

            # mean and var from REST
            json_payload = self._build_json_payload(python_domain, python_cov, historical_data, points_to_evaluate.tolist())
            resp = self.testapp.post(self.endpoint, json_payload)
            resp_schema = GpMeanVarResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_mean = numpy.asarray(resp_dict.get('mean'))
            rest_var = numpy.asarray(resp_dict.get('var'))

            self.assert_vector_within_relative(rest_mean, cpp_mean, tolerance)
            self.assert_vector_within_relative(rest_var, cpp_var, tolerance)

    def _compare_endpoint_mean_var_results(
            self,
            json_payload,
            endpoint,
            response_schema,
            tolerance,
            truth_mean=None,
            truth_var=None
    ):
        """Compare the results of mean/var endpoint to truth values.

        :param json_payload: json input to POST to the test endpoint
        :type json_payload: string (json)
        :param endpoint: path to the endpoint to test
        :type endpoint: string
        :param response_schema: schema of the endpoint's response
        :type response_schema: colander.MappingSchema subclass
        :param tolerance: desired relative accuracy
        :type tolerance: float64
        :param truth_mean: the true GP mean values
        :type truth_mean: array with shape matching the response schema 'mean' field
        :param truth_var: the true GP variance values
        :type truth_var: array with shape matching the response schema 'var' field

        """
        resp = self.testapp.post(endpoint, json_payload)
        resp_dict = response_schema.deserialize(json.loads(resp.body))

        if truth_mean is not None:
            rest_mean = numpy.asarray(resp_dict.get('mean'))
            self.assert_vector_within_relative(rest_mean, truth_mean, tolerance)
        if truth_var is not None:
            rest_var = numpy.asarray(resp_dict.get('var'))
            self.assert_vector_within_relative(rest_var, truth_var, tolerance)

    def test_interfaces_equivalent(self):
        """Test that the /gp/mean, var, mean_var, etc. endpoints are consistent."""
        tolerance = numpy.finfo(numpy.float64).eps
        for test_case in self.gp_test_environments:
            python_domain, python_gp = test_case
            python_cov, historical_data = python_gp.get_core_data_copy()

            points_to_evaluate = python_domain.generate_uniform_random_points_in_domain(10)

            # mean and var from REST
            json_payload = self._build_json_payload(python_domain, python_cov, historical_data, points_to_evaluate.tolist())
            resp = self.testapp.post(self.endpoint, json_payload)
            resp_schema = GpMeanVarResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))

            truth_mean = numpy.asarray(resp_dict.get('mean'))
            truth_var = numpy.asarray(resp_dict.get('var'))

            self._compare_endpoint_mean_var_results(
                json_payload,
                GP_MEAN_ENDPOINT,
                GpMeanResponse(),
                tolerance,
                truth_mean=truth_mean,
            )

            self._compare_endpoint_mean_var_results(
                json_payload,
                GP_VAR_ENDPOINT,
                GpVarResponse(),
                tolerance,
                truth_var=truth_var,
            )

            self._compare_endpoint_mean_var_results(
                json_payload,
                GP_VAR_DIAG_ENDPOINT,
                GpVarDiagResponse(),
                tolerance,
                truth_var=numpy.diag(truth_var),
            )

            self._compare_endpoint_mean_var_results(
                json_payload,
                GP_MEAN_VAR_DIAG_ENDPOINT,
                GpMeanVarDiagResponse(),
                tolerance,
                truth_mean=truth_mean,
                truth_var=numpy.diag(truth_var),
            )


if __name__ == "__main__":
    T.run()
