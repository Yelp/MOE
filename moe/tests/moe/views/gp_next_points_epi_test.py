# -*- coding: utf-8 -*-
"""Test class for gp_next_points_epi view."""
import simplejson as json
import testify as T

from moe.tests.moe.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.views.gp_next_points_pretty_view import GpNextPointsResponse


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

    endpoints = [
            '/gp/next_points/epi',
            '/gp/next_points/kriging',
            '/gp/next_points/constant_liar',
            ]

    def _build_json_payload(self, GP, num_samples_to_generate):
        """Create a json_payload to POST to the /gp/next_points/* endpoint with all needed info."""
        json_payload = json.dumps({
            'num_samples_to_generate': num_samples_to_generate,
            'gp_info': self._build_gp_info(GP),
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/next_points/* endpoints do the same thing as the C++ interface."""
        for endpoint in self.endpoints:
            for test_case in self.test_cases:
                num_points_in_sample = test_case['num_points_in_sample']
                num_samples_to_generate = test_case['num_samples_to_generate']
                domain = test_case['domain']

                GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)

                # Next point from REST
                json_payload = self._build_json_payload(GP, num_samples_to_generate)
                resp = self.testapp.post(endpoint, json_payload)
                resp_schema = GpNextPointsResponse()
                resp_dict = resp_schema.deserialize(json.loads(resp.body))
                T.assert_in('points_to_sample', resp_dict)
                T.assert_in('expected_improvement', resp_dict)

                for ei in resp_dict['expected_improvement']:
                    T.assert_gte(ei, 0.0)

if __name__ == "__main__":
    T.run()
