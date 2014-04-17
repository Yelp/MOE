"""A class to encapsulate 'pretty' views for gp_next_points_* endpoints.

Include:
    1. Request and response schemas
    2. Class that extends GpPrettyView for next_points optimizers
"""

import colander
from moe.views.schemas import GpInfo, EiOptimizationParameters, ListOfPointsInDomain, ListOfExpectedImprovements
from moe.optimal_learning.EPI.src.python.constant import default_ei_optimization_parameters

from moe.views.gp_pretty_view import GpPrettyView
from moe.views import _make_gp_from_gp_info

import moe.build.GPP as C_GP
from moe.optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import ExpectedImprovementOptimizationParameters


class GpNextPointsRequest(colander.MappingSchema):

    """A gp_next_points_* request colander schema."""

    num_samples_to_generate = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()
    ei_optimization_parameters = EiOptimizationParameters(
            missing=default_ei_optimization_parameters._asdict(),
            )


class GpNextPointsResponse(colander.MappingSchema):

    """A gp_next_points_* response colander schema."""

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = ListOfExpectedImprovements()


class GpNextPointsPrettyView(GpPrettyView):

    """A class to encapsulate 'pretty' gp_next_points_* views.

    Extends GpPrettyView with:
        1. GP generation from params
        2. Converting params into a C++ consumable set of optimization parameters
        3. A method (compute_next_points_to_sample_response) for computing the next best points to sample from a GP

    """

    request_schema = GpNextPointsRequest()
    response_schema = GpNextPointsResponse()

    pretty_default_request = {
            "num_samples_to_generate": 1,
            "gp_info": GpPrettyView.pretty_default_gp_info,
            }

    def compute_next_points_to_sample_response(self, params, optimization_method_name, route_name, *args, **kwargs):
        """Compute the next points to sample (and their expected improvement) using optimization_method_name from params in the request."""
        num_samples_to_generate = params.get('num_samples_to_generate')

        GP = self.make_gp(params)
        ei_optimization_parameters_cpp = self.get_optimization_parameters_cpp(params)

        optimization_method = getattr(GP, optimization_method_name)

        next_points = optimization_method(
                ei_optimization_parameters_cpp,
                num_samples_to_generate,
                *args,
                **kwargs
                )
        expected_improvement = GP.evaluate_expected_improvement_at_point_list(next_points)

        json_points_to_sample = list([list(row) for row in next_points])

        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': json_points_to_sample,
                'expected_improvement': list(expected_improvement),
                })

    @staticmethod
    def make_gp(params):
        """Create a GP object from params."""
        gp_info = params.get('gp_info')
        return _make_gp_from_gp_info(gp_info)

    @staticmethod
    def get_optimization_parameters_cpp(params):
        """Form a C++ conumable ExpectedImprovementOptimizationParameters object from params."""
        ei_optimization_parameters = params.get('ei_optimization_parameters')

        return ExpectedImprovementOptimizationParameters(
                domain_type=C_GP.DomainTypes.tensor_product,
                optimizer_type=C_GP.OptimizerTypes.gradient_descent,
                num_random_samples=0,
                optimizer_parameters=C_GP.GradientDescentParameters(
                    ei_optimization_parameters.get('num_multistarts'),
                    ei_optimization_parameters.get('gd_iterations'),
                    ei_optimization_parameters.get('max_num_restarts'),
                    ei_optimization_parameters.get('gamma'),
                    ei_optimization_parameters.get('pre_mult'),
                    ei_optimization_parameters.get('max_relative_change'),
                    ei_optimization_parameters.get('tolerance'),
                    ),
                )
