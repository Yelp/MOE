# -*- coding: utf-8 -*-
"""Simple functions for hitting the REST endpoints of a MOE bandit service."""
import simplejson as json

from moe.easy_interface import call_endpoint_with_payload, DEFAULT_HOST, DEFAULT_PORT
from moe.views.constant import ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT, BANDIT_BLA_ROUTE_NAME
from moe.views.schemas.bandit_pretty_view import BanditResponse


def bandit(
        historical_info,
        type=BANDIT_BLA_ROUTE_NAME,
        rest_host=DEFAULT_HOST,
        rest_port=DEFAULT_PORT,
        testapp=None,
        **kwargs
        ):
    """Hit the rest endpoint for allocating arms of a bandit given arms already sampled (historical info)."""
    endpoint = ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[type]
    raw_payload = kwargs.copy()
    raw_payload['historical_info'] = historical_info.json_payload()

    json_payload = json.dumps(raw_payload)

    json_response = call_endpoint_with_payload(rest_host, rest_port, endpoint, json_payload, testapp)

    output = BanditResponse().deserialize(json_response)

    return output['arm_allocations']
