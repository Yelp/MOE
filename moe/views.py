from pyramid.response import Response
from pyramid.view import view_config

import simplejson as json

import build.GPP as C_GP
from optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import OptimalGaussianProcessLinkedCpp, ExpectedImprovementOptimizationParameters
from optimal_learning.EPI.src.python.models.covariance_of_process import CovarianceOfProcess
from optimal_learning.EPI.src.python.models.sample_point import SamplePoint
from optimal_learning.EPI.src.python.constant import EIOptimizationParameters, GaussianProcessParameters, default_ei_optimization_parameters, default_gaussian_process_parameters

from moe.schemas import GpMeanVarRequest, GpEiRequest, GpEiResponse, GpMeanVarResponse, GpNextPointsEpiRequest, GpNextPointsEpiResponse
from moe.constant import pretty_default

@view_config(route_name='home', renderer='moe:templates/index.mako')
def index_page(request):
    return {
            'nav_active': 'home',
            }

@view_config(route_name='gp_plot', renderer='moe:templates/gp_plot.mako')
def gp_plot_page(request):
    return {
            'nav_active': 'demo',
            'default_gaussian_process_parameters': default_gaussian_process_parameters,
            'default_ei_optimization_parameters': default_ei_optimization_parameters,
            }

# Pretty inputs

@view_config(route_name='gp_next_points_epi_pretty', renderer='moe:templates/pretty_input.mako')
def gp_next_points_epi_pretty_view(request):
    return {
            'endpoint': 'gp_next_points_epi',
            'default_text': json.dumps(pretty_default['gp_next_points_epi']),
            }

@view_config(route_name='gp_ei_pretty', renderer='moe:templates/pretty_input.mako')
def gp_ei_pretty_view(request):
    return {
            'endpoint': 'gp_ei',
            'default_text': json.dumps(pretty_default['gp_ei']),
            }

@view_config(route_name='gp_mean_var_pretty', renderer='moe:templates/pretty_input.mako')
def gp_mean_var_pretty_view(request):
    return {
            'endpoint': 'gp_mean_var',
            'default_text': json.dumps(pretty_default['gp_mean_var']),
            }

# GP routes

def _make_default_covariance_of_process(signal_variance=None, length=None):
    """Make a default covariance of process with optional parameters
    """
    hyperparameters = [signal_variance]
    hyperparameters.extend(length)

    return CovarianceOfProcess(hyperparameters=hyperparameters)

def _make_gp_from_gp_info(gp_info):
    """Create and return a C++ backed GP from a gp_info dict

    gp_info has the form of GpInfo in moe/schemas.py

    """
    # Load up the info
    points_sampled = gp_info['points_sampled']
    domain = gp_info['domain']
    signal_variance = gp_info['signal_variance']
    length = gp_info['length_scale']

    # Build the required objects
    covariance_of_process = _make_default_covariance_of_process(
            signal_variance=signal_variance,
            length=length,
            )
    GP = OptimalGaussianProcessLinkedCpp(
            domain=domain,
            covariance_of_process=covariance_of_process,
            )

    # Sample from the process
    for point in points_sampled:
        sample_point = SamplePoint(
                point['point'],
                point['value'],
                point['value_var'],
                )
        GP.add_sample_point(sample_point, point['value_var'])

    return GP

@view_config(route_name='gp_next_points_epi', renderer='json', request_method='POST')
def gp_next_points_epi_view(request):
    params_schema = GpNextPointsEpiRequest()
    params = params_schema.deserialize(request.json_body)

    num_samples_to_generate = params.get('num_samples_to_generate')
    gp_info = params.get('gp_info')
    
    ei_optimization_parameters_dict = params.get('ei_optimization_parameters')
    ei_optimization_parameters = EIOptimizationParameters(**ei_optimization_parameters_dict)

    GP = _make_gp_from_gp_info(gp_info)

    ei_optimization_parameters_cpp = ExpectedImprovementOptimizationParameters(
            domain_type=C_GP.DomainTypes.tensor_product,
            optimizer_type=C_GP.OptimizerTypes.gradient_descent,
            num_random_samples=0,
            optimizer_parameters=C_GP.GradientDescentParameters(
                ei_optimization_parameters.num_multistarts,
                ei_optimization_parameters.gd_iterations,
                ei_optimization_parameters.max_num_restarts,
                ei_optimization_parameters.gamma,
                ei_optimization_parameters.pre_mult,
                ei_optimization_parameters.max_relative_change,
                ei_optimization_parameters.tolerance,
                ),
            )
    next_points = GP.multistart_expected_improvement_optimization(
            ei_optimization_parameters_cpp,
            num_samples_to_generate,
            )
    expected_improvement = GP.evaluate_expected_improvement_at_point_list(next_points)

    json_points_to_sample = list([list(row) for row in next_points])

    resp_schema = GpNextPointsEpiResponse()
    resp = resp_schema.serialize({
            'endpoint': 'gp_next_points_epi',
            'points_to_sample': json_points_to_sample,
            'expected_improvement': list(expected_improvement),
            })

    return resp

@view_config(route_name='gp_mean_var', renderer='json', request_method='POST')
def gp_mean_var_view(request):
    params_schema = GpMeanVarRequest()
    params = params_schema.deserialize(request.json_body)

    points_to_sample = params.get('points_to_sample')
    gp_info = params.get('gp_info')

    GP = _make_gp_from_gp_info(gp_info)

    mean, var = GP.get_mean_and_var_of_points(points_to_sample)

    json_var = list([list(row) for row in var])

    resp_schema = GpMeanVarResponse()
    resp = resp_schema.serialize({
            'endpoint': 'gp_mean_var',
            'mean': list(mean),
            'var': json_var,
            })

    return resp

@view_config(route_name='gp_ei', renderer='json', request_method='POST')
def gp_ei_view(request):
    params_schema = GpEiRequest()
    params = params_schema.deserialize(request.json_body)

    points_to_evaluate = params.get('points_to_evaluate')
    points_being_sampled = params.get('points_being_sampled')
    mc_iterations = params.get('mc_iterations')

    gp_info = params.get('gp_info')
    if not gp_info:
        raise(ValueError, "POST request to /gp/ei needs gp_info")

    GP = _make_gp_from_gp_info(gp_info)

    expected_improvement = GP.evaluate_expected_improvement_at_point_list(
            points_to_evaluate,
            points_being_sampled=points_being_sampled,
            )

    resp_schema = GpEiResponse()
    resp = resp_schema.serialize({
            'endpoint': 'gp_ei',
            'expected_improvement': expected_improvement,
            })

    return resp
