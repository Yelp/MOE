"""Various tests for checking exceptions in views."""
import testify as T
from webtest.app import AppError

from moe.tests.moe.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase


class RestGaussianProcessTestCaseWithExceptions(RestGaussianProcessTestCase):

    """Test that proper errors are thrown when endpoints get universally bad data."""

    endpoints = [
            '/gp/ei',
            '/gp/mean_var',
            '/gp/next_points/epi',
            '/gp/next_points/kriging',
            '/gp/next_points/constant_liar',
            ]

    def test_empty_json_payload_invalid(self):
        """Test empty json payload causes an AppError."""
        for endpoint in self.endpoints:
            T.assert_raises(AppError, self.testapp.post, endpoint, '{}')

    def test_badly_formed_json_payload_invalid(self):
        """Test malformed json payload causes a ValueError."""
        for endpoint in self.endpoints:
            T.assert_raises(ValueError, self.testapp.post, endpoint, '}')
