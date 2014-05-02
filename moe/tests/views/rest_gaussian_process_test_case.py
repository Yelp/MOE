# -*- coding: utf-8 -*-
"""Base class for testing the REST interface against the C++ interface."""
import testify as T

from moe.tests.optimal_learning.python.OLD_gaussian_process_test_case import OLDGaussianProcessTestCase


class RestGaussianProcessTestCase(OLDGaussianProcessTestCase):

    """Base class for testing the REST interface against the C++ interface."""

    endpoint = None

    domain_1d = [[0, 1]]
    domain_2d = [[0, 1], [0, 1]]
    domain_3d = [[0, 1], [0, 1], [0, 1]]

    @T.class_setup
    def create_webapp(self):
        """Create a mocked webapp and store it in self.testapp."""
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)

    @staticmethod
    def _build_gp_info(GP):
        """Create and return a gp_info dictionary from a GP object."""
        # Convert sampled points
        json_points_sampled = []
        for i, point in enumerate(GP.points_sampled):
            json_points_sampled.append({
                    'point': point.point.tolist(),  # json needs the numpy array to be a list
                    'value': point.value,
                    'value_var': GP.sample_variance_of_samples[i],
                    })

        # Build entire gp_info dict
        gp_info = {
                'points_sampled': json_points_sampled,
                'domain': GP.domain,
                }

        return gp_info
