# -*- coding: utf-8 -*-
"""Various tests for checking exceptions in views."""
import testify as T
from webtest.app import AppError

from moe.views.constant import ALL_REST_MOE_ROUTES
from moe.tests.moe.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase


class RestGaussianProcessTestCaseWithExceptions(RestGaussianProcessTestCase):

    """Test that proper errors are thrown when endpoints get universally bad data."""

    def test_empty_json_payload_invalid(self):
        """Test empty json payload causes an AppError."""
        for moe_route in ALL_REST_MOE_ROUTES:
            T.assert_raises(AppError, self.testapp.post, moe_route.endpoint, '{}')

    def test_badly_formed_json_payload_invalid(self):
        """Test malformed json payload causes a ValueError."""
        for moe_route in ALL_REST_MOE_ROUTES:
            T.assert_raises(ValueError, self.testapp.post, moe_route.endpoint, '}')
