# -*- coding: utf-8 -*-
"""Test class for gp_next_points_epi view."""
import simplejson as json
import testify as T

from moe.tests.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.views.gp_next_points_pretty_view import GpNextPointsResponse
from moe.views.constant import ALL_NEXT_POINTS_MOE_ROUTES, GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME


class TestGpNextPointsViews(RestGaussianProcessTestCase):

    """Test that the /gp/next_points/* endpoints do the same thing as the C++ interface."""

    test_cases = [
            {
                'domain': RestGaussianProcessTestCase.domain_1d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_2d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_3d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            ]

    def _build_json_payload(self, GP, num_samples_to_generate, lie_value=None):
        """Create a json_payload to POST to the /gp/next_points/* endpoint with all needed info."""
        dict_to_dump = {
            'num_samples_to_generate': num_samples_to_generate,
            'gp_info': self._build_gp_info(GP),
            }
        if lie_value is not None:
            dict_to_dump['lie_value'] = lie_value
        return json.dumps(dict_to_dump)

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/next_points/* endpoints do the same thing as the C++ interface."""
        for moe_route in ALL_NEXT_POINTS_MOE_ROUTES:
            for test_case in self.test_cases:
                num_points_in_sample = test_case['num_points_in_sample']
                num_samples_to_generate = test_case['num_samples_to_generate']
                domain = test_case['domain']

                GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)

                # Next point from REST
                if moe_route.route_name == GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME:
                    json_payload = self._build_json_payload(GP, num_samples_to_generate, lie_value=0.0)
                else:
                    json_payload = self._build_json_payload(GP, num_samples_to_generate)
                resp = self.testapp.post(moe_route.endpoint, json_payload)
                resp_schema = GpNextPointsResponse()
                resp_dict = resp_schema.deserialize(json.loads(resp.body))
                T.assert_in('points_to_sample', resp_dict)
                T.assert_in('expected_improvement', resp_dict)

                for ei in resp_dict['expected_improvement']:
                    T.assert_gte(ei, 0.0)

if __name__ == "__main__":
    T.run()
