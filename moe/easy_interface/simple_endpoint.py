# -*- coding: utf-8 -*-
"""Simple functions for hitting the REST endpoints of a MOE service."""
from future import standard_library
standard_library.install_aliases()
import contextlib
import urllib.request, urllib.error, urllib.parse

import simplejson as json

from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.views.constant import ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT, GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_MEAN_VAR_ROUTE_NAME, GP_HYPER_OPT_ROUTE_NAME
from moe.views.schemas.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.schemas.rest.gp_hyper_opt import GpHyperOptResponse
from moe.views.schemas.rest.gp_mean_var import GpMeanVarResponse


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6543


def call_endpoint_with_payload(rest_host, rest_port, endpoint, json_payload, testapp=None):
    """Send a POST request to a ``url`` with a given ``json_payload``, return the response as a dict."""
    if testapp is None:
        url = "http://{0}:{1:d}{2}".format(rest_host, rest_port, endpoint)
        request = urllib.request.Request(url, json_payload, {'Content-Type': 'application/json'})
        with contextlib.closing(urllib.request.urlopen(request)) as f:
            response = f.read()
    else:
        response = testapp.post(endpoint, json_payload).body

    return json.loads(response)


def gp_next_points(
        moe_experiment,
        method_route_name=GP_NEXT_POINTS_EPI_ROUTE_NAME,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        testapp=None,
        **kwargs
):
    """Hit the rest endpoint for finding next point of highest EI at rest_host:rest_port corresponding to the method with the given experiment."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[method_route_name]
    raw_payload = kwargs.copy()  # Any options can be set via the kwargs ('covariance_info' etc.)

    experiment_payload = moe_experiment.build_json_payload()

    if 'gp_historical_info' not in raw_payload:
        raw_payload['gp_historical_info'] = experiment_payload.get('gp_historical_info')

    if 'domain_info' not in raw_payload:
        raw_payload['domain_info'] = experiment_payload.get('domain_info')

    json_payload = json.dumps(raw_payload)

    json_response = call_endpoint_with_payload(rest_host, rest_port, endpoint, json_payload, testapp)

    output = GpNextPointsResponse().deserialize(json_response)

    return output["points_to_sample"]


def gp_hyper_opt(
        points_sampled,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        testapp=None,
        **kwargs
        ):
    """Hit the rest endpoint for optimizing the hyperparameters of a gaussian process, given points already sampled."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[GP_HYPER_OPT_ROUTE_NAME]
    # This will fail if len(points_sampled) == 0; but then again this endpoint doesn't make sense with 0 historical data
    gp_dim = len(points_sampled[0][0])
    raw_payload = kwargs.copy()

    # Sanitize input points
    points_sampled_clean = [SamplePoint._make(point) for point in points_sampled]
    historical_data = HistoricalData(
            gp_dim,
            sample_points=points_sampled_clean,
            )

    if 'domain_info' not in raw_payload:
        raw_payload['domain_info'] = {'dim': gp_dim}

    if 'gp_historical_info' not in raw_payload:
        raw_payload['gp_historical_info'] = historical_data.json_payload()

    if 'hyperparameter_domain_info' not in raw_payload:
        hyper_dim = gp_dim + 1  # default covariance has this many parameters
        raw_payload['hyperparameter_domain_info'] = {
            'dim': hyper_dim,
            'domain_bounds': [{'min': 0.1, 'max': 2.0}] * hyper_dim,
        }

    json_payload = json.dumps(raw_payload)

    json_response = call_endpoint_with_payload(rest_host, rest_port, endpoint, json_payload, testapp)

    output = GpHyperOptResponse().deserialize(json_response)

    return output['covariance_info']


def gp_mean_var(
        points_sampled,
        points_to_evaluate,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        testapp=None,
        **kwargs
):
    """Hit the rest endpoint for calculating the posterior mean and variance of a gaussian process, given points already sampled."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[GP_MEAN_VAR_ROUTE_NAME]
    raw_payload = kwargs.copy()  # Any options can be set via the kwargs ('covariance_info' etc.)

    raw_payload['points_to_evaluate'] = points_to_evaluate

    # Sanitize input points
    points_sampled_clean = [SamplePoint._make(point) for point in points_sampled]
    historical_data = HistoricalData(
            len(points_to_evaluate[0]),  # The dim of the space
            sample_points=points_sampled_clean,
            )

    if 'gp_historical_info' not in raw_payload:
        raw_payload['gp_historical_info'] = historical_data.json_payload()

    if 'domain_info' not in raw_payload:
        raw_payload['domain_info'] = {'dim': len(points_to_evaluate[0])}

    json_payload = json.dumps(raw_payload)

    json_response = call_endpoint_with_payload(rest_host, rest_port, endpoint, json_payload, testapp)

    output = GpMeanVarResponse().deserialize(json_response)

    return output.get('mean'), output.get('var')
