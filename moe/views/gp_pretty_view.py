# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views."""
import simplejson as json

PRETTY_RENDERER = 'moe:templates/pretty_input.mako'


class GpPrettyView(object):

    """A class to encapsulate 'pretty' views.

    These views have:
        1. A backend endpoint
        2. A pretty, browser interactable view with forms to test the backend endpoint

    """

    _route_name = None
    _pretty_route_name = None

    request_schema = None  # Define in a subclass
    response_schema = None  # Define in a subclass

    _pretty_default_request = None
    _pretty_default_gp_info = {
            "points_sampled": [
                {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                {"value_var": 0.01, "value": 0.2, "point": [1.0]},
                ],
            "domain": [[0, 1]],
            }

    def __init__(self, request):
        """Store the request for the view."""
        self.request = request

    def pretty_response(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.
        
        :returns: A dictionary with 'endpoint' and 'default_text' keys.
        
        """
        return {
                'endpoint': self._route_name,
                'default_text': json.dumps(self._pretty_default_request),
                }

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.
        
        :returns: A deserialized self.request_schema object

        """
        return self.request_schema.deserialize(self.request.json_body)

    def form_response(self, response_dict):
        """Return the serialized response object from a dict.
        
        :param response_dict: a dict that can be serialized by self.response_schema
        :type response_dict: dict
        :returns: a serialized self.response_schema object
        """
        return self.response_schema.serialize(response_dict)
