# -*- coding: utf-8 -*-
"""Base class for testing the REST interface against the C++ interface."""
import testify as T

from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase


class RestGaussianProcessTestCase(GaussianProcessTestCase):

    """Base class for testing the REST interface against the C++ interface."""

    endpoint = None

    @T.class_setup
    def create_webapp(self):
        """Create a mocked webapp and store it in self.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)

    @staticmethod
    def _build_covariance_info(covariance):
        """Create and return a covariance_info dictionary from a :class:`~moe.optimal_learning.python.python_version.covaraince.Covaraince` object."""
        return {
                'covariance_type': covariance.covariance_type,
                'hyperparameters': covariance.get_hyperparameters().tolist(),
                }

    @staticmethod
    def _build_gp_info(gaussian_process):
        """Create and return a gp_info dictionary from a GP object."""
        # Convert sampled points
        json_points_sampled = []
        for point in gaussian_process._historical_data.to_list_of_sample_points():
            json_points_sampled.append({
                    'point': point.point.tolist(),  # json needs the numpy array to be a list
                    'value': point.value,
                    'value_var': point.noise_variance,
                    })

        # Build entire gp_info dict
        gp_info = {
                'points_sampled': json_points_sampled,
                }

        return gp_info
