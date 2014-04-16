"""Classes for gp_next_points_epi endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""

from pyramid.view import view_config

from moe.views import _make_gp_from_gp_info
from moe.views.gp_pretty_view import GpPrettyView

import colander
from moe.views.schemas import GpInfo, EiOptimizationParameters, ListOfPointsInDomain, ListOfExpectedImprovements
from moe.optimal_learning.EPI.src.python.constant import default_ei_optimization_parameters

import moe.build.GPP as C_GP
from moe.optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import ExpectedImprovementOptimizationParameters


class GpNextPointsEpiRequest(colander.MappingSchema):

    """A gp_next_points_epi request colander schema."""

    num_samples_to_generate = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()
    ei_optimization_parameters = EiOptimizationParameters(
            missing=default_ei_optimization_parameters._asdict(),
            )


class GpNextPointsEpiResponse(colander.MappingSchema):

    """A gp_next_points_epi response colander schema."""

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = ListOfExpectedImprovements()


class GpNextPointsEpi(GpPrettyView):

    """Views for gp_next_points_epi endpoints."""

    route_name = 'gp_next_points_epi'
    pretty_route_name = 'gp_next_points_epi_pretty'

    request_schema = GpNextPointsEpiRequest()
    response_schema = GpNextPointsEpiResponse()

    pretty_default_request = {
            "num_samples_to_generate": 1,
            "gp_info": GpPrettyView.pretty_default_gp_info,
            }

    @view_config(route_name=pretty_route_name, renderer=GpPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactable view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_next_points_epi_view(self):
        """Endpoint for gp_next_points_epi POST requests."""
        params = self.get_params_from_request()

        num_samples_to_generate = params.get('num_samples_to_generate')
        gp_info = params.get('gp_info')

        ei_optimization_parameters = params.get('ei_optimization_parameters')

        GP = _make_gp_from_gp_info(gp_info)

        ei_optimization_parameters_cpp = ExpectedImprovementOptimizationParameters(
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
        next_points = GP.multistart_expected_improvement_optimization(
                ei_optimization_parameters_cpp,
                num_samples_to_generate,
                )
        expected_improvement = GP.evaluate_expected_improvement_at_point_list(next_points)

        json_points_to_sample = list([list(row) for row in next_points])

        return self.form_response({
                'endpoint': 'gp_next_points_epi',
                'points_to_sample': json_points_to_sample,
                'expected_improvement': list(expected_improvement),
                })
