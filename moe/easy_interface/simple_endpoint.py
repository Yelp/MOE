# -*- coding: utf-8 -*-
"""Simple functions for hitting the REST endpoints of a MOE service."""
import simplejson as json
import urllib2

from moe.views.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.rest.gp_mean_var import GpMeanVarResponse
from moe.views.constant import ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT, GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_MEAN_VAR_ROUTE_NAME


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6543


def call_endpoint_with_payload(url, json_payload):
    request = urllib2.Request(url, json_payload, {'Content-Type': 'application/json'})
    f = urllib2.urlopen(request)
    response = f.read()
    f.close()

    return json.loads(response)

def gp_next_points(
        moe_experiment,
        num_to_sample=1,
        method_route_name=GP_NEXT_POINTS_EPI_ROUTE_NAME,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
):
    """Hit the rest endpoint for finding next point of highest EI at rest_host:rest_port corresponding to the method with the given experiment."""
    raw_payload = kwargs.copy()
    experiment_payload = moe_experiment.build_json_payload()
    raw_payload['gp_info'] = experiment_payload.get('gp_info')
    raw_payload['domain_info'] = experiment_payload.get('domain_info')
    raw_payload['num_to_sample'] = num_to_sample

    json_payload = json.dumps(raw_payload)

    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[method_route_name]

    url = "http://%s:%d%s" % (rest_host, rest_port, endpoint)

    json_response = call_endpoint_with_payload(url, json_payload)

    output = GpNextPointsResponse().deserialize(json_response)

    return output["points_to_sample"]

def gp_mean_var(
        points_sampled,
        points_to_sample,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
):
    """Hit the rest endpoint for calculating the posterior mean and variance of a gaussian process, given points already sampled."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[GP_MEAN_VAR_ROUTE_NAME]
    raw_payload = kwargs.copy()
    raw_payload['points_to_sample'] = points_to_sample
    json_ready_points_sampled = []
    for point in points_sampled:
        json_ready_points_sampled.append({
            'point': point[0],
            'value': point[1],
            'value_var': point[2] if len(point) == 3 else 0.0,
            })
    raw_payload['gp_info'] = {
            'points_sampled': json_ready_points_sampled,
            }
    raw_payload['domain_info'] = {'dim': 1}
    json_payload = json.dumps(raw_payload)
    url = "http://%s:%d%s" % (rest_host, rest_port, endpoint)

    print url
    print json_payload

    json_response = call_endpoint_with_payload(url, json_payload)

    output = GpMeanVarResponse().deserialize(json_response)

    return output
