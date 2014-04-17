# -*- coding: utf-8 -*-
"""Test class for gp_mean_var view."""
import testify as T
import simplejson as json

from moe.tests.moe.views.rest_gaussian_process_test_case import RestGaussianProcessTestCase
from moe.optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points
from moe.optimal_learning.EPI.src.python.constant import default_expected_improvement_parameters

from moe.views.gp_ei import GpEiResponse


class TestGpEiView(RestGaussianProcessTestCase):

    """Test that the /gp/ei endpoint does the same thing as the C++ interface."""

    test_cases = [
            {
                'domain': RestGaussianProcessTestCase.domain_1d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_1d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_2d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_2d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': RestGaussianProcessTestCase.domain_3d,
                'points_to_evaluate': list([list(p) for p in get_latin_hypercube_points(10, RestGaussianProcessTestCase.domain_3d)]),
                'num_points_in_sample': 10,
                },
            ]

    def _build_json_payload(self, GP, points_to_evaluate):
        """Create a json_payload to POST to the /gp/ei endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_evaluate': points_to_evaluate,
            'gp_info': self._build_gp_info(GP),
            'points_being_sampled': [],
            'mc_iterations': default_expected_improvement_parameters.mc_iterations,
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
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
            resp = self.testapp.post('/gp/ei', json_payload)
            resp_schema = GpEiResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_expected_improvement = resp_dict.get('expected_improvement')

            self.assert_lists_relatively_equal(
                    cpp_expected_improvement,
                    rest_expected_improvement,
                    tol=1e-11,
                    )

if __name__ == "__main__":
    T.run()
