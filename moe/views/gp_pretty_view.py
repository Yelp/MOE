"""A class to encapsulate 'pretty' views."""
import simplejson as json


class GpPrettyView(object):

    """A class to encapsulate 'pretty' views.

    These views have:
        1. A backend endpoint
        2. A pretty, browser interactable view with forms to test the backend endpoint

    """

    route_name = None
    pretty_route_name = None

    request_schema = None
    response_schema = None

    pretty_default_request = None
    pretty_default_gp_info = {
            "points_sampled": [
                {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                {"value_var": 0.01, "value": 0.2, "point": [1.0]},
                ],
            "domain": [[0, 1]],
            }
    pretty_renderer = 'moe:templates/pretty_input.mako'

    def __init__(self, request):
        """Store the request for the view."""
        self.request = request

    def pretty_response(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return {
                'endpoint': self.route_name,
                'default_text': json.dumps(self.pretty_default_request),
                }

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request."""
        return self.request_schema.deserialize(self.request.json_body)

    def form_response(self, response_dict):
        """Return the serialized response object from a dict."""
        return self.response_schema.serialize(response_dict)
