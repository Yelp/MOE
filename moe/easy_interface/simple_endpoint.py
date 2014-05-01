# -*- coding: utf-8 -*-
"""Simple functions for hitting the REST endpoints of a MOE service."""
import simplejson as json
import urllib2

from moe.views.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.constant import ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT, GP_NEXT_POINTS_EPI_ROUTE_NAME


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6543


def gp_next_points(
        moe_experiment,
        num_samples_to_generate=1,
        method_route_name=GP_NEXT_POINTS_EPI_ROUTE_NAME,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        **kwargs
):
    """Hit the rest endpoint for finding next point of highest EI at rest_host:rest_port corresponding to the method with the given experiment."""
    raw_payload = kwargs.copy()
    raw_payload['gp_info'] = moe_experiment.__dict__()
    raw_payload['num_samples_to_generate'] = num_samples_to_generate

    json_payload = json.dumps(raw_payload)

    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[method_route_name]
    url = "http://%s:%d%s" % (rest_host, rest_port, endpoint)

    request = urllib2.Request(url, json_payload, {'Content-Type': 'application/json'})
    f = urllib2.urlopen(request)
    response = f.read()
    f.close()

    json_response = json.loads(response)
    output = GpNextPointsResponse().deserialize(json_response)
    return output["points_to_sample"]
