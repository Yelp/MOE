# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views for gp_next_points_* endpoints.

Include:
    1. Request and response schemas
    2. Class that extends GpPrettyView for next_points optimizers
"""
import colander
import numpy

from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement, multistart_expected_improvement_optimization
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import GpInfo, ListOfPointsInDomain, ListOfExpectedImprovements, CovarianceInfo, BoundedDomainInfo, OptimizationInfo, OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES
from moe.views.utils import _make_gp_from_params, _make_domain_from_params
from moe.optimal_learning.python.linkers import OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS
from moe.optimal_learning.python.constant import OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS, TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS


class GpNextPointsRequest(colander.MappingSchema):

    """A gp_next_points_* request colander schema.

    **Required fields**

        :gp_info: a :class:`moe.views.schemas.GpInfo` object of historical data

    **Optional fields**

        :num_samples_to_generate: number of next points to generate (default: 1)
        :ei_optimization_parameters: moe.views.schemas.EiOptimizationParameters() object containing optimization parameters (default: moe.optimal_learning.python.constant.default_ei_optimization_parameters)

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'num_samples_to_generate': 1,
            'gp_info': {
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                },
            },
        }

    """

    num_samples_to_generate = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()
    covariance_info = CovarianceInfo()
    domain_info = BoundedDomainInfo()
    optimization_info = OptimizationInfo()


class GpNextPointsResponse(colander.MappingSchema):

    """A gp_next_points_* response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :points_to_sample: list of points in the domain to sample next (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :expected_improvement: list of EI of points in points_to_sample (:class:`moe.views.schemas.ListOfExpectedImprovements`)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "expected_improvement": ["0.443478498868"],
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = ListOfExpectedImprovements()


class GpNextPointsPrettyView(GpPrettyView):

    """A class to encapsulate 'pretty' gp_next_points_* views.

    Extends GpPrettyView with:
        1. gaussian_process generation from params
        2. Converting params into a C++ consumable set of optimization parameters
        3. A method (compute_next_points_to_sample_response) for computing the next best points to sample from a gaussian_process

    """

    request_schema = GpNextPointsRequest()
    response_schema = GpNextPointsResponse()

    ei_optimization_method = multistart_expected_improvement_optimization

    _pretty_default_request = {
            "num_samples_to_generate": 1,
            "gp_info": GpPrettyView._pretty_default_gp_info,
            }

    def compute_next_points_to_sample_response(self, params, optimization_method_name, route_name, *args, **kwargs):
        """Compute the next points to sample (and their expected improvement) using optimization_method_name from params in the request.

        :param deserialized_request_params: the deserialized REST request, containing ei_optimization_parameters and gp_info
        :type deserialized_request_params: a deserialized self.request_schema object as a dict
        :param optimization_method_name: the optimization method to use
        :type optimization_method_name: string in ``moe.views.constant.OPTIMIZATION_METHOD_NAMES``
        :param route_name: name of the route being called
        :type route_name: string in ``moe.views.constant.ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT.keys()``
        :param *args: extra args to be passed to optimization method
        :param **kwargs: extra kwargs to be passed to optimization method

        """
        num_samples_to_generate = params.get('num_samples_to_generate')
        points_being_sampled = params.get('points_being_sampled')
        if points_being_sampled is not None:
            points_being_sampled = numpy.array(points_being_sampled)

        gaussian_process = _make_gp_from_params(params)
        domain = _make_domain_from_params(params)

        optimizer_class, optimization_parameters, num_random_samples = self.get_optimization_parameters_cpp(params)

        expected_improvement_evaluator = ExpectedImprovement(
                gaussian_process,
                points_to_sample=points_being_sampled,
                num_mc_iterations=TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
                )

        expected_improvement_optimizer = optimizer_class(
                domain,
                expected_improvement_evaluator,
                optimization_parameters,
                num_random_samples=num_random_samples,
                )

        next_points = multistart_expected_improvement_optimization(
                expected_improvement_optimizer,
                optimization_parameters.num_multistarts,
                num_samples_to_generate,
                )

        expected_improvement = expected_improvement_evaluator.evaluate_at_point_list(
                next_points,
                )

        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': next_points.tolist(),
                'expected_improvement': expected_improvement.tolist(),
                })

    @staticmethod
    def get_optimization_parameters_cpp(deserialized_request_params):
        """Figure out which cpp_wrappers.* objects to construct from params.

        :param deserialized_request_params: the deserialized REST request, containing ei_optimization_parameters
        :type deserialized_request_params: a dictionary with a key ei_optimization_parameters containing a :class:`moe.views.schemas.EiOptimizationParameters()` object with optimization parameters

        """
        optimization_info = deserialized_request_params.get('optimization_info')
        num_random_samples = optimization_info.get('num_random_samples')

        optimization_method = OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS[optimization_info.get('optimization_type')]
        schema_class = OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES[optimization_info.get('optimization_type')]()

        # Start with defaults
        optimization_parameters_dict = dict(OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS[optimization_info.get('optimization_type')]._asdict())
        for param, val in optimization_info.get('optimization_parameters', {}).iteritems():
            # Override defaults as needed
            optimization_parameters_dict[param] = val

        # Validate optimization parameters
        validated_optimization_parameters = schema_class.deserialize(optimization_parameters_dict)
        validated_optimization_parameters['num_multistarts'] = optimization_info['num_multistarts']
        optimization_parameters = optimization_method.cpp_parameters_class(**validated_optimization_parameters)

        return optimization_method.cpp_optimizer_class, optimization_parameters, num_random_samples
