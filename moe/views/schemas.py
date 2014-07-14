# -*- coding: utf-8 -*-
"""Base level schemas for the response/request schemas of each MOE REST endpoint."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_NEWTON_PARAMETERS, DEFAULT_GRADIENT_DESCENT_PARAMETERS, GRADIENT_DESCENT_OPTIMIZER, DEFAULT_OPTIMIZATION_MULTISTARTS, DEFAULT_OPTIMIZATION_NUM_RANDOM_SAMPLES, TENSOR_PRODUCT_DOMAIN_TYPE, SQUARE_EXPONENTIAL_COVARIANCE_TYPE, NULL_OPTIMIZER, NEWTON_OPTIMIZER, DOMAIN_TYPES, OPTIMIZATION_TYPES, COVARIANCE_TYPES, CONSTANT_LIAR_METHODS, DEFAULT_MAX_NUM_THREADS, DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, CONSTANT_LIAR_MIN, LIKELIHOOD_TYPES, LOG_MARGINAL_LIKELIHOOD


class PositiveFloat(colander.SchemaNode):

    """Colander positive (finite) float."""

    schema_type = colander.Float
    title = 'Positive Float'

    def validator(self, node, cstruct):
        """Raise an exception if the node value (cstruct) is non-positive or non-finite.

        :param node: the node being validated (usually self)
        :type node: colander.SchemaNode subclass instance
        :param cstruct: the value being validated
        :type cstruct: float
        :raise: colander.Invalid if cstruct value is bad

        """
        if not 0.0 < cstruct < float('inf'):
            raise colander.Invalid(node, msg='Value = {0:f} must be positive and finite.'.format(cstruct))


class ListOfPositiveFloats(colander.SequenceSchema):

    """Colander list of positive floats."""

    float_in_list = PositiveFloat()


class ListOfFloats(colander.SequenceSchema):

    """Colander list of floats."""

    float_in_list = colander.SchemaNode(colander.Float())


class SinglePoint(colander.MappingSchema):

    """A point object.

    Contains:

        * point - ListOfFloats
        * value - float
        * value_var - float >= 0.0

    """

    point = ListOfFloats()
    value = colander.SchemaNode(colander.Float())
    value_var = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            missing=0.0,
            )


class PointsSampled(colander.SequenceSchema):

    """A list of SinglePoint objects."""

    point_sampled = SinglePoint()


class DomainCoordinate(colander.MappingSchema):

    """A single domain interval."""

    min = colander.SchemaNode(colander.Float())
    max = colander.SchemaNode(colander.Float())


class Domain(colander.SequenceSchema):

    """A list of domain interval DomainCoordinate objects."""

    domain_coordinates = DomainCoordinate()


class DomainInfo(colander.MappingSchema):

    """The domain info needed for every request.

    **Required fields**

        :dim: the dimension of the domain (int)

    **Optional fields**

        :domain_type: the type of domain to use in ``moe.optimal_learning.python.python_version.constant.DOMAIN_TYPES`` (default: TENSOR_PRODUCT_DOMAIN_TYPE)

    """

    domain_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(DOMAIN_TYPES),
            missing=TENSOR_PRODUCT_DOMAIN_TYPE,
            )
    dim = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=0),
            )


class BoundedDomainInfo(DomainInfo):

    """The domain info needed for every request, along with bounds for optimization.

    **Required fields**

        All required fields from :class:`~moe.views.schemas.DomainInfo`
        :domain_bounds: the bounds of the domain of type :class:`moe.views.schemas.Domain`

    """

    domain_bounds = Domain()


class GradientDescentParametersSchema(colander.MappingSchema):

    """Parameters for the gradient descent optimizer.

    See :class:`moe.optimal_learning.python.cpp_wrappers.optimization.GradientDescentParameters`

    """

    max_num_steps = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.max_num_steps,
            validator=colander.Range(min=1),
            )
    max_num_restarts = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.max_num_restarts,
            validator=colander.Range(min=1),
            )
    num_steps_averaged = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.num_steps_averaged,
            validator=colander.Range(min=1),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.gamma,
            validator=colander.Range(min=0.0),
            )
    pre_mult = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.pre_mult,
            validator=colander.Range(min=0.0),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_GRADIENT_DESCENT_PARAMETERS.max_relative_change,
            validator=colander.Range(
                min=0.0,
                max=1.0,
                ),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_NEWTON_PARAMETERS.tolerance,
            validator=colander.Range(min=0.0),
            )


class NewtonParametersSchema(colander.MappingSchema):

    """Parameters for the newton optimizer.

    See :class:`moe.optimal_learning.python.cpp_wrappers.optimization.NewtonParameters`

    """

    max_num_steps = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_NEWTON_PARAMETERS.max_num_steps,
            validator=colander.Range(min=1),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_NEWTON_PARAMETERS.gamma,
            validator=colander.Range(min=0.0),
            )
    time_factor = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_NEWTON_PARAMETERS.time_factor,
            validator=colander.Range(min=0.0),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_NEWTON_PARAMETERS.max_relative_change,
            validator=colander.Range(
                min=0.0,
                max=1.0,
                ),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_NEWTON_PARAMETERS.tolerance,
            validator=colander.Range(min=0.0),
            )


class NullParametersSchema(colander.MappingSchema):

    """Parameters for the null optimizer."""

    pass


class CovarianceInfo(colander.MappingSchema):

    """The covariance info needed for every request.

    **Required fields**

        :covariance_type: a covariance type in ``moe.optimal_learning.python.python_version.constant.COVARIANCE_TYPES``
        :hyperparameters: the hyperparameters corresponding to the given covariance_type

    """

    covariance_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(COVARIANCE_TYPES),
            missing=SQUARE_EXPONENTIAL_COVARIANCE_TYPE,
            )
    # TODO(GH-216): Improve hyperparameter validation. All > 0 is ok for now but eventually individual covariance objects should
    # provide their own validation.
    hyperparameters = ListOfPositiveFloats(
            missing=None,
            )


class GpHistoricalInfo(colander.MappingSchema):

    """The Gaussian Process info needed for every request.

    Contains:

        * points_sampled - PointsSampled

    """

    points_sampled = PointsSampled()


class ListOfPointsInDomain(colander.SequenceSchema):

    """A list of lists of floats."""

    point_in_domain = ListOfFloats()


class ListOfExpectedImprovements(colander.SequenceSchema):

    """A list of floats all geq 0.0."""

    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0),
            )


class MatrixOfFloats(colander.SequenceSchema):

    """A 2d list of floats."""

    row_of_matrix = ListOfFloats()


OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES = {
        NULL_OPTIMIZER: NullParametersSchema,
        NEWTON_OPTIMIZER: NewtonParametersSchema,
        GRADIENT_DESCENT_OPTIMIZER: GradientDescentParametersSchema,
        }


class OptimizationInfo(colander.MappingSchema):

    """Optimization information needed for each next point endpoint.

    **Optimization fields**

        :optimization_type: a string defining the optimization type from `moe.optimal_learning.python.constant.OPTIMIZATION_TYPES` (default: GRADIENT_DESCENT_OPTIMIZER)
        :optimization_parameters: a dict corresponding the the parameters of the optimization method

    """

    optimization_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(OPTIMIZATION_TYPES),
            missing=GRADIENT_DESCENT_OPTIMIZER,
            )
    num_multistarts = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_OPTIMIZATION_MULTISTARTS,
            validator=colander.Range(min=1),
            )
    num_random_samples = colander.SchemaNode(
            colander.Int(),
            missing=DEFAULT_OPTIMIZATION_NUM_RANDOM_SAMPLES,
            validator=colander.Range(min=1),
            )


class GpNextPointsRequest(colander.MappingSchema):

    """A ``gp_next_points_*`` request colander schema.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :mc_iterations: number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
        :max_num_threads: maximum number of threads to use in computation (default: 1)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimization_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information
        :points_being_sampled: list of points in domain being sampled in concurrent experiments (default: [])

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "points_being_sampled": [[0.2], [0.7]],
            "mc_iterations": 10000,
            "max_num_threads": 1,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "domain_type": "tensor_product"
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
            "optimization_info": {
                "optimization_type": "gradient_descent_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimization_parameters": {
                    "gamma": 0.5,
                    ...
                    },
                },
        }

    """

    num_to_sample = colander.SchemaNode(
            colander.Int(),
            missing=1,
            validator=colander.Range(min=1),
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            )
    max_num_threads = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = BoundedDomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )
    optimization_info = OptimizationInfo(
            missing=OptimizationInfo().deserialize({}),
            )
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )


class GpNextPointsConstantLiarRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.gp_next_points_pretty_view.GpNextPointsRequest` with a lie value.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :lie_method: a string from `CONSTANT_LIAR_METHODS` representing the liar method to use (default: 'constant_liar_min')
        :lie_value: a float representing the 'lie' the Constant Liar heuristic will use (default: None). If `lie_value` is not None the algorithm will use this value instead of one calculated using `lie_method`.
        :lie_noise_variance: a positive (>= 0) float representing the noise variance of the 'lie' value (default: 0.0)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimiaztion_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "lie_value": 0.0,
            "lie_noise_variance": 0.0,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    """

    lie_method = colander.SchemaNode(
            colander.String(),
            missing=CONSTANT_LIAR_MIN,
            validator=colander.OneOf(CONSTANT_LIAR_METHODS),
            )
    lie_value = colander.SchemaNode(
            colander.Float(),
            missing=None,
            )
    lie_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.gp_next_points_pretty_view.GpNextPointsRequest` with kriging parameters.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :std_deviation_coef: a float used in Kriging, see Kriging implementation docs (default: 0.0)
        :kriging_noise_variance: a positive (>= 0) float used in Kriging, see Kriging implementation docs (default: 0.0)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimiaztion_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "std_deviation_coef": 0.0,
            "kriging_noise_variance": 0.0,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    """

    std_deviation_coef = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            )
    kriging_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsResponse(colander.MappingSchema):

    """A ``gp_next_points_*`` response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :points_to_sample: list of points in the domain to sample next (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :expected_improvement: list of EI of points in points_to_sample (:class:`moe.views.schemas.ListOfExpectedImprovements`)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "expected_improvement": "0.443478498868",
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )


class GpHyperOptRequest(colander.MappingSchema):

    """A gp_hyper_opt request colander schema.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` object of historical data
        :domain_info: a :class:`moe.views.schemas.DomainInfo` dict of domain information for the GP
        :hyperparameter_domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information for the hyperparameter optimization

    **Optional fields**

        :max_num_threads: maximum number of threads to use in computation (default: 1)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information, used as a starting point for optimization
        :optimization_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "max_num_threads": 1,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
            "hyperparameter_domain_info": {
                "dim": 2,
                "domain_bounds": [
                    {"min": 0.1, "max": 2.0},
                    {"min": 0.1, "max": 2.0},
                    ],
                },
            "optimization_info": {
                "optimization_type": "gradient_descent_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimization_parameters": {
                    "gamma": 0.5,
                    ...
                    },
                },
            "log_likelihood_info": "log_marginal_likelihood"
        }

    """

    max_num_threads = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )
    hyperparameter_domain_info = BoundedDomainInfo()
    optimization_info = OptimizationInfo(
            missing=OptimizationInfo().deserialize({}),
            )
    log_likelihood_info = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(LIKELIHOOD_TYPES),
            missing=LOG_MARGINAL_LIKELIHOOD,
            )


class GpHyperOptStatus(colander.MappingSchema):

    """A gp_hyper_opt status schema.

    **Output fields**

       :log_likelihood: The log likelihood at the new hyperparameters
       :grad_log_likelihood: The gradient of the log likelihood at the new hyperparameters
       :optimization_success: Whether or not the optimizer converged to an optimal set of hyperparameters

    """

    log_likelihood = colander.SchemaNode(colander.Float())
    grad_log_likelihood = ListOfFloats()
    optimization_success = colander.SchemaNode(colander.String())


class GpHyperOptResponse(colander.MappingSchema):

    """A gp_hyper_opt response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_hyper_opt",
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [0.88, 1.24],
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    covariance_info = CovarianceInfo()
    status = GpHyperOptStatus()


class GpMeanVarRequest(colander.MappingSchema):

    """A gp_mean_var request colander schema.

    **Required fields**

        :points_to_sample: list of points in domain to calculate the Gaussian Process (GP) mean and covariance at (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` object of historical data

    **Optional fields**

        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_sample": [[0.1], [0.5], [0.9]],
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_sample": [[0.1], [0.5], [0.9]],
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "domain_type": "tensor_product"
                "dim": 1,
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
        }

    """

    points_to_sample = ListOfPointsInDomain()
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )


class GpEndpointResponse(colander.MappingSchema):

    """A base schema for the endpoint name.

    **Output fields**

        :endpoint: the endpoint that was called

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_mean_var",
        }

    """

    endpoint = colander.SchemaNode(colander.String())


class GpMeanMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the mean of a gaussian process.

    **Output fields**

        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "mean": ["0.0873832198661","0.0130505261903","0.174755506336"],
        }

    """

    mean = ListOfFloats()


class GpVarMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the [co]variance of a gaussian process.

    **Output fields**

        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": [
                    ["0.228910114429","0.0969433771923","0.000268292907969"],
                    ["0.0969433771923","0.996177332647","0.0969433771923"],
                    ["0.000268292907969","0.0969433771923","0.228910114429"]
                ],
        }

    """

    var = MatrixOfFloats()


class GpVarDiagMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the variance of a gaussian process.

    **Output fields**

        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": ["0.228910114429","0.996177332647","0.228910114429"],
        }

    """

    var = ListOfFloats()


class GpMeanResponse(GpEndpointResponse, GpMeanMixinResponse):

    """A gp_mean response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpVarResponse(GpEndpointResponse, GpVarMixinResponse):

    """A gp_var response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpVarDiagResponse(GpEndpointResponse, GpVarDiagMixinResponse):

    """A gp_var_diag response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpMeanVarResponse(GpMeanResponse, GpVarMixinResponse):

    """A gp_mean_var response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)
        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpMeanVarDiagResponse(GpMeanResponse, GpVarDiagMixinResponse):

    """A gp_mean_var_diag response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)
        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

    See composing members' docstrings.

    """

    pass
