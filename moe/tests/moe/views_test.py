# -*- coding: utf-8 -*-
import testify as T
import simplejson as json

from moe.tests.EPI.src.python.gaussian_process_test_case import GaussianProcessTestCase
from moe.optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points
from moe.optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import ExpectedImprovementOptimizationParameters
from moe.optimal_learning.EPI.src.python.constant import default_expected_improvement_parameters, default_ei_optimization_parameters
import moe.build.GPP as C_GP

from moe.schemas import GpEiResponse, GpMeanVarResponse, GpNextPointsEpiResponse

class RestGaussianProcessTestCase(GaussianProcessTestCase):

    """Base class for testing the REST interface against the C++ interface."""

    @T.class_setup
    def create_webapp(self):
        from moe import main
        app = main({}, use_mongo='false')
        from webtest import TestApp
        self.testapp = TestApp(app)

    @staticmethod
    def _build_gp_info(GP):
        """Create and return a gp_info dictionary from a GP object."""

        # Convert sampled points
        json_points_sampled = []
        for i, point in enumerate(GP.points_sampled):
            json_points_sampled.append({
                    'point': list(point.point), # json needs the numpy array to be a list
                    'value': point.value,
                    'value_var': GP.sample_variance_of_samples[i],
                    })

        # Build entire gp_info dict
        gp_info = {
                'points_sampled': json_points_sampled,
                'domain': GP.domain,
                }

        return gp_info

class TestGpMeanVarView(RestGaussianProcessTestCase):

    """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""
    domain_1d = [[0,1]]
    domain_2d = [[0,1],[0,1]]
    domain_3d = [[0,1],[0,1],[0,1]]
    test_cases = [
            {
                'domain': domain_1d,
                'points_to_sample': list([list(p) for p in get_latin_hypercube_points(10, domain_1d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_2d,
                'points_to_sample': list([list(p) for p in get_latin_hypercube_points(10, domain_2d)]),
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_3d,
                'points_to_sample': list([list(p) for p in get_latin_hypercube_points(10, domain_3d)]),
                'num_points_in_sample': 10,
                },
            ]

    def _build_json_payload(self, GP, points_to_sample):
        """Create a json_payload to POST to the /gp/mean_var endpoint with all needed info."""
        json_payload = json.dumps({
            'points_to_sample': points_to_sample,
            'gp_info': self._build_gp_info(GP),
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/mean_var endpoint does the same thing as the C++ interface."""
        for test_case in self.test_cases:
            points_to_sample = test_case['points_to_sample']
            num_points_in_sample = test_case['num_points_in_sample']
            domain = test_case['domain']

            GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)
            # EI from C++
            cpp_mean, cpp_var = GP.get_mean_and_var_of_points(points_to_sample)

            # EI from REST
            json_payload = self._build_json_payload(GP, points_to_sample)
            resp = self.testapp.post('/gp/mean_var', json_payload)
            resp_schema = GpMeanVarResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            rest_mean = resp_dict.get('mean')
            rest_var = resp_dict.get('var')

            self.assert_lists_relatively_equal(rest_mean, cpp_mean, tol=1e-11)
            self.assert_matrix_relatively_equal(rest_var, cpp_var, tol=1e-11)

class TestGpNextPointsEiView(RestGaussianProcessTestCase):

    """Test that the /gp/ei endpoint does the same thing as the C++ interface."""

    domain_1d = [[0,1]]
    domain_2d = [[0,1],[0,1]]
    domain_3d = [[0,1],[0,1],[0,1]]
    test_cases = [
            {
                'domain': domain_1d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_2d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            {
                'domain': domain_3d,
                'num_samples_to_generate': 1,
                'num_points_in_sample': 10,
                },
            ]

    def _build_json_payload(self, GP, num_samples_to_generate):
        """Create a json_payload to POST to the /gp/next_points/epi endpoint with all needed info."""
        json_payload = json.dumps({
            'num_samples_to_generate': num_samples_to_generate,
            'gp_info': self._build_gp_info(GP),
            })
        return json_payload

    def test_interface_returns_same_as_cpp(self):
        """Test that the /gp/next_points/epi endpoint does the same thing as the C++ interface."""
        for test_case in self.test_cases:
            num_points_in_sample = test_case['num_points_in_sample']
            num_samples_to_generate = test_case['num_samples_to_generate']
            domain = test_case['domain']

            GP, _ = self._make_random_processes_from_latin_hypercube(domain, num_points_in_sample)
            # Next points from C++

            ei_optimization_parameters = ExpectedImprovementOptimizationParameters(
                    domain_type=C_GP.DomainTypes.tensor_product,
                    optimizer_type=C_GP.OptimizerTypes.gradient_descent,
                    num_random_samples=0,
                    optimizer_parameters=C_GP.GradientDescentParameters(
                        default_ei_optimization_parameters.num_multistarts,
                        default_ei_optimization_parameters.gd_iterations,
                        default_ei_optimization_parameters.max_num_restarts,
                        default_ei_optimization_parameters.gamma,
                        default_ei_optimization_parameters.pre_mult,
                        default_ei_optimization_parameters.max_relative_change,
                        default_ei_optimization_parameters.tolerance,
                        ),
                    )
            next_points = GP.multistart_expected_improvement_optimization(
                    ei_optimization_parameters,
                    num_samples_to_generate,
                    )

            # Next point from REST
            json_payload = self._build_json_payload(GP, num_samples_to_generate)
            resp = self.testapp.post('/gp/next_points/epi', json_payload)
            resp_schema = GpNextPointsEpiResponse()
            resp_dict = resp_schema.deserialize(json.loads(resp.body))
            T.assert_in('points_to_sample', resp_dict)
            T.assert_in('expected_improvement', resp_dict)

            for ei in resp_dict['expected_improvement']:
                T.assert_gte(ei, 0.0)

class TestGpEiView(RestGaussianProcessTestCase):

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
