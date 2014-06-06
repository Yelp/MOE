# -*- coding: utf-8 -*-
"""Simple functions for hitting the REST endpoints of a MOE service."""
import urllib2

import simplejson as json

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.views.constant import ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT, GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_MEAN_VAR_ROUTE_NAME, GP_HYPER_OPT_ROUTE_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.rest.gp_hyper_opt import GpHyperOptResponse
from moe.views.rest.gp_mean_var import GpMeanVarResponse


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6543


def call_endpoint_with_payload(url, json_payload):
    """Send a POST request to a ``url`` with a given ``json_payload``, return the response as a dict."""
    request = urllib2.Request(url, json_payload, {'Content-Type': 'application/json'})
    f = urllib2.urlopen(request)
    response = f.read()
    f.close()

    return json.loads(response)


def gp_next_points(
        moe_experiment,
        method_route_name=GP_NEXT_POINTS_EPI_ROUTE_NAME,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
):
    """Hit the rest endpoint for finding next point of highest EI at rest_host:rest_port corresponding to the method with the given experiment."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[method_route_name]
    raw_payload = kwargs.copy()  # Any options can be set via the kwargs ('covariance_info' etc.)

    experiment_payload = moe_experiment.build_json_payload()
    if 'gp_info' not in raw_payload:
        raw_payload['gp_info'] = experiment_payload.get('gp_info')
    if 'domain_info' not in raw_payload:
        raw_payload['domain_info'] = experiment_payload.get('domain_info')

    json_payload = json.dumps(raw_payload)

    url = "http://{0}:{1:d}{2}".format(rest_host, rest_port, endpoint)

    json_response = call_endpoint_with_payload(url, json_payload)

    output = GpNextPointsResponse().deserialize(json_response)

    return output["points_to_sample"]


def _build_gp_info_from_points_sampled(points_sampled):
    json_ready_points_sampled = []
    for point in points_sampled:
        json_ready_points_sampled.append({
            'point': point[0].tolist(),
            'value': point[1],
            'value_var': point[2] if len(point) == 3 else 0.0,
            })
    return {
        'points_sampled': json_ready_points_sampled,
        }


def gp_hyper_opt(
        points_sampled,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
        ):
    """Hit the rest endpoint for optimizing the hyperparameters of a gaussian process, given points already sampled."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[GP_HYPER_OPT_ROUTE_NAME]
    gp_dim = len(points_sampled[0][0])
    hyper_dim = gp_dim + 1
    raw_payload = kwargs.copy()
    raw_payload['domain_info'] = {'dim': gp_dim}
    raw_payload['gp_info'] = _build_gp_info_from_points_sampled(points_sampled)
    raw_payload['hyperparameter_domain_info'] = {
            'dim': hyper_dim,
            'domain_bounds': [],
            }
    for _ in range(hyper_dim):
        raw_payload['hyperparameter_domain_info']['domain_bounds'].append({
            'min': 0.1,
            'max': 2.0,
            })
    json_payload = json.dumps(raw_payload)

    url = "http://{0}:{1:d}{2}".format(rest_host, rest_port, endpoint)

    json_response = call_endpoint_with_payload(url, json_payload)

    output = GpHyperOptResponse().deserialize(json_response)

    return output.get('covariance_info')


def gp_mean_var(
        points_sampled,
        points_to_sample,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
):
    """Hit the rest endpoint for calculating the posterior mean and variance of a gaussian process, given points already sampled."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[GP_MEAN_VAR_ROUTE_NAME]
    raw_payload = kwargs.copy()  # Any options can be set via the kwargs ('covariance_info' etc.)

    raw_payload['points_to_sample'] = points_to_sample

    historical_data = HistoricalData(
            len(points_to_sample[0]),  # The dim of the space
            sample_points=points_sampled,
            )
    if 'gp_info' not in raw_payload:
        raw_payload['gp_info'] = historical_data.json_payload()
    if 'domain_info' not in raw_payload:
        raw_payload['domain_info'] = {'dim': len(points_to_sample[0])}

    json_payload = json.dumps(raw_payload)

    url = "http://{0}:{1:d}{2}".format(rest_host, rest_port, endpoint)

    json_response = call_endpoint_with_payload(url, json_payload)

    output = GpMeanVarResponse().deserialize(json_response)

    return output.get('mean'), output.get('var')
