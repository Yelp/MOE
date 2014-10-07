# -*- coding: utf-8 -*-
"""Various tests for checking exceptions in views."""
import pytest

import copy
import logging

import colander

import simplejson as json

from webtest.app import AppError

from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.tests.views.rest_test_case import RestTestCase
from moe.views.constant import ALL_REST_MOE_ROUTES, GP_MEAN_VAR_ENDPOINT, GP_NEXT_POINTS_EPI_ENDPOINT
from moe.views.exceptions import general_error, failed_colander_validation
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.rest.gp_mean_var import GpMeanVarView
from moe.views.schemas.gp_next_points_pretty_view import GpNextPointsRequest
from moe.views.schemas.rest.gp_mean_var import GpMeanVarRequest
from moe.views.utils import _make_gp_from_params


class TestRestGaussianProcessWithExceptions(GaussianProcessTestCase, RestTestCase):

    """Test that proper errors are thrown when endpoints bad data."""

    @pytest.fixture(autouse=True)
    def disable_logging(self):
        """Disable logging (for the duration of this test case)."""
        logging.disable(logging.CRITICAL)

        def fin():
            """Re-enable logging (so other test cases are unaffected)."""
            logging.disable(logging.NOTSET)

    def test_empty_json_payload_invalid(self):
        """Test empty json payload causes an AppError."""
        for moe_route in ALL_REST_MOE_ROUTES:
            with pytest.raises(AppError):
                self.testapp.post(moe_route.endpoint, {})

    def test_badly_formed_json_payload_invalid(self):
        """Test malformed json payload causes a ValueError."""
        truth_result = self.testapp.post(GP_MEAN_VAR_ENDPOINT, '}', expect_errors=True)
        for moe_route in ALL_REST_MOE_ROUTES:
            test_result = self.testapp.post(moe_route.endpoint, '}', expect_errors=True)
            assert truth_result.body == test_result.body

    def test_invalid_hyperparameters_input(self):
        """Test that invalid hyperparameters (via GP_MEAN_VAR_ENDPOINT) generate expected Response with error message."""
        endpoint = GP_MEAN_VAR_ENDPOINT
        dict_payload = copy.deepcopy(GpMeanVarView._pretty_default_request)

        # Invalidate a hyperparameter
        dict_payload['covariance_info']['hyperparameters'][0] *= -1.0
        result = self.testapp.post(endpoint, json.dumps(dict_payload), expect_errors=True)

        # Get the colander exception that arises from processing invalid hyperparameters
        request_schema = GpMeanVarRequest()
        try:
            request_schema.deserialize(dict_payload)
        except colander.Invalid as request_exception:
            pass

        assert result.body == failed_colander_validation(request_exception, result.request).body

    def test_invalid_points_sampled_input(self):
        """Test that duplicate points_sampled (via GP_NEXT_POINTS_EPI_ENDPOINT) generate expected Response with error message."""
        endpoint = GP_NEXT_POINTS_EPI_ENDPOINT
        dict_payload = copy.deepcopy(GpNextPointsPrettyView._pretty_default_request)

        # Invalidate historical info: 0.0 noise and add a duplicate point
        for sample_point in dict_payload['gp_historical_info']['points_sampled']:
            sample_point['value_var'] = 0.0

        dict_payload['gp_historical_info']['points_sampled'].append(dict_payload['gp_historical_info']['points_sampled'][0])
        result = self.testapp.post(endpoint, json.dumps(dict_payload), expect_errors=True)

        # Get the exception that arises from processing invalid hyperparameters
        request_schema = GpNextPointsRequest()
        params = request_schema.deserialize(dict_payload)
        try:
            _make_gp_from_params(params)
        except Exception as request_exception:
            pass

        assert result.body == general_error(request_exception, result.request).body
