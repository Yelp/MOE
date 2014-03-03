# -*- coding: utf-8 -*-

import testify as T
import urllib2
import json

from tests.EPI.src.python.gaussian_process_test_case import GaussianProcessTestCase
from optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points

class TestGpEiView(GaussianProcessTestCase):

    """Test that the /gp/ei endpoint does the same thing as the C++ interface."""

    domain_1d = [[0,1]]
    domain_2d = [[0,1],[0,1]]
    domain_3d = [[0,1],[0,1],[0,1]]
    test_cases = [
            {
                'domain': domain_1d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, domain_1d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_2d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, domain_2d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_3d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, domain_3d)]),
                'num_points_in_sample': 10,
                },
            ]

    @staticmethod
    def _build_json_payload(GP, points_to_evaluate):
        """Create a json_payload to POST to the /gp/ei endpoint with all needed info."""
        json_points_sampled = []
        for i, point in enumerate(GP.points_sampled):
            json_points_sampled.append({
                    'point': list(point.point), # json needs the numpy array to be a list
                    'value': point.value,
                    'value_var': GP.sample_variance_of_samples[i],
                    })
        json_payload = json.dumps({
            'points_to_evaluate': points_to_evaluate,
            'gp_info': {
                'points_sampled': json_points_sampled,
                'domain': GP.domain,
                },
            })
        return json_payload

    def test_integration(self):
        """Test that the /gp/ei endpoint does the same thing as the C++ interface."""
        for test_case in self.test_cases:
            points_to_evaluate = test_case['points_to_evaluate']
            num_points_in_sample = test_case['num_points_in_sample']
            domain = test_case['domain']

            GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)
            # EI from C++
            cpp_expected_improvement = GP.evaluate_expected_improvement_at_point_list(
                    points_to_evaluate,
                    )

            # EI from REST
            json_payload = self._build_json_payload(GP, points_to_evaluate)
            headers = {'Content-Type':'application/json; charset=utf-8'}
            req = urllib2.Request('http://localhost:6543/gp/ei', json_payload, headers)
            resp = urllib2.urlopen(req)
            resp_json = json.loads(resp.read())
            rest_expected_improvement = resp_json['expected_improvement']

            for i, cpp_ei_at_point in enumerate(cpp_expected_improvement):
                rest_ei_at_point = rest_expected_improvement[i]
                self.assert_relatively_equal(cpp_ei_at_point, rest_ei_at_point)

if __name__ == "__main__":
    T.run()
