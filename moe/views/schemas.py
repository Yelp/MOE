# -*- coding: utf-8 -*-
"""Base level schemas for the response/request schemas of each MOE REST endpoint.

.. Warning:: Outputs of colander schema serialization/deserialization should be treated as
  READ-ONLY. It appears that "missing=" and "default=" value are weak-copied (by reference).
  Thus changing missing/default fields in the output dict can modify the schema!

TODO(GH-291): make sure previous warning is moved to the schemas/__init__.py file

"""
import colander

from moe.optimal_learning.python.constant import GRADIENT_DESCENT_OPTIMIZER, L_BFGS_B_OPTIMIZER, TENSOR_PRODUCT_DOMAIN_TYPE, SQUARE_EXPONENTIAL_COVARIANCE_TYPE, NULL_OPTIMIZER, NEWTON_OPTIMIZER, DOMAIN_TYPES, OPTIMIZER_TYPES, COVARIANCE_TYPES, CONSTANT_LIAR_METHODS, DEFAULT_MAX_NUM_THREADS, MAX_ALLOWED_NUM_THREADS, DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, LIKELIHOOD_TYPES, LOG_MARGINAL_LIKELIHOOD, DEFAULT_CONSTANT_LIAR_METHOD, DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE, DEFAULT_KRIGING_NOISE_VARIANCE, DEFAULT_KRIGING_STD_DEVIATION_COEF


class StrictMappingSchema(colander.MappingSchema):

    """A ``colander.MappingSchema`` that raises exceptions when asked to serialize/deserialize unknown keys.

    .. Note:: by default, colander.MappingSchema ignores/throws out unknown keys.

    """

    def schema_type(self, **kw):
        """Set MappingSchema to raise ``colander.Invalid`` when serializing/deserializing unknown keys.

        This overrides the staticmethod of the same name in ``colander._SchemaNode``.
        ``schema_type`` encodes the same information as the ``typ`` ctor argument to
        ``colander.SchemaNode``
        See: http://colander.readthedocs.org/en/latest/api.html#colander.SchemaNode

        .. Note:: Passing ``typ`` or setting ``schema_type`` in subclasses will ***override*** this!

        This solution follows: https://github.com/Pylons/colander/issues/116

        .. Note:: colander's default behavior is ``unknown='ignore'``; the other option
          is ``'preserve'``. See: http://colander.readthedocs.org/en/latest/api.html#colander.Mapping

        """
        return colander.Mapping(unknown='raise')


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


class SinglePoint(StrictMappingSchema):

    """A point object.

    **Required fields**

    :ivar point: (:class:`moe.views.schemas.ListOfFloats`) The point sampled (in the domain of the function)
    :ivar value: (*float64*) The value returned by the function
    :ivar value_var: (*float64 >= 0.0*) The noise/measurement variance (if any) associated with :attr:`value`

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


class DomainCoordinate(StrictMappingSchema):

    """A single domain interval."""

    min = colander.SchemaNode(colander.Float())
    max = colander.SchemaNode(colander.Float())


class Domain(colander.SequenceSchema):

    """A list of domain interval DomainCoordinate objects."""

    domain_coordinates = DomainCoordinate()


class DomainInfo(StrictMappingSchema):

    """The domain info needed for every request.

    **Required fields**

    :ivar dim: (*int >= 0*) the dimension of the domain (int)

    **Optional fields**

    :ivar domain_type: (*str*) the type of domain to use, one of :const:`moe.optimal_learning.python.python_version.constant.DOMAIN_TYPES` (default: TENSOR_PRODUCT_DOMAIN_TYPE)

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

    .. Note:: For EI/next points, selecting a domain that is substantially larger than
      the bounding box of the historical data may lead MOE to favor exploring near the
      boundaries instead of near existing data.

    **Required fields**

    All required fields from :class:`~moe.views.schemas.DomainInfo`

    :ivar domain_bounds: (*list of list of float64*) the bounds of the domain of type :class:`moe.views.schemas.Domain`

    """

    domain_bounds = Domain()


class GradientDescentParametersSchema(StrictMappingSchema):

    """Parameters for the gradient descent optimizer.

    See :class:`moe.optimal_learning.python.cpp_wrappers.optimization.GradientDescentParameters`

    """

    max_num_steps = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    max_num_restarts = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    num_steps_averaged = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=0),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )
    pre_mult = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0, max=1.0),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )


class LBFGSBParametersSchema(StrictMappingSchema):

    """Parameters for the L-BFGS-B optimizer.

    See :class:`moe.optimal_learning.python.cpp_wrappers.optimization.GradientDescentParameters`

    """

    approx_grad = colander.SchemaNode(
            colander.Boolean(),
            )
    max_func_evals = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    max_metric_correc = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    factr = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=1.0),
            )
    pgtol = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )
    epsilon = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )


class NewtonParametersSchema(StrictMappingSchema):

    """Parameters for the newton optimizer.

    See :class:`moe.optimal_learning.python.cpp_wrappers.optimization.NewtonParameters`

    """

    max_num_steps = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=1.0),
            )
    time_factor = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=1.0e-16),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0, max=1.0),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )


class NullParametersSchema(StrictMappingSchema):

    """Parameters for the null optimizer."""

    pass


class CovarianceInfo(StrictMappingSchema):

    """The covariance info needed for every request.

    .. Warning:: Very large length scales (adverse conditioning effects) and very small length scales (irrelevant dimensions)
      can negatively impact MOE's performance. It may be worth checking that your length scales are "reasonable."

      Additionally, MOE's default optimization parameters were tuned for hyperparameter values roughly in [0.01, 100].
      Venturing too far out of this range means the defaults may perform poorly.

    **Required fields**

    :ivar covariance_type: (*str*) a covariance type in :const:`moe.optimal_learning.python.python_version.constant.COVARIANCE_TYPES`
    :ivar hyperparameters: (*list of float64*) the hyperparameters corresponding to the given :attr:`covariance_type`

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


class GpHistoricalInfo(StrictMappingSchema):

    """The Gaussian Process info needed for every request.

    .. Warning:: If the points are too close together (relative to the length scales in :class:`moe.views.schemas.CovarianceInfo`)
      with simultaneously very low or zero noise, the condition number of the GPP's covariance matrix can be very large. The
      matrix may even become numerically singular.

      In such cases, check for (nearly) duplicates points and be mindful of large length scales.

    .. Warning:: 0 ``noise_variance`` in the input historical data may lead to [numerically] singular covariance matrices. This
      becomes more likely as ``num_sampled`` increases. Noise caps the condition number at roughly ``1.0 / min(noise)``, so
      adding artificial noise (e.g., ``1.0e-12``) can aid with conditioning issues.

      MOE does not do this for you automatically since 0 noise may be extremely important for some users.

    .. Note:: MOE performs best if the input ``points_sampled_value`` are 0 mean.

    **Required fields**

    :ivar points_sampled: (*list of PointsSampled*) The :class:`moe.views.schemas.PointsSampled` (point, value, noise) that make up
      the historical data.

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


#: Mapping from optimizer types (:const:`moe.optimal_learning.python.constant.OPTIMIZER_TYPES`) to
#: optimizer schemas, e.g., :class:`moe.views.schemas.NewtonParametersSchema`.
OPTIMIZER_TYPES_TO_SCHEMA_CLASSES = {
        NULL_OPTIMIZER: NullParametersSchema,
        NEWTON_OPTIMIZER: NewtonParametersSchema,
        GRADIENT_DESCENT_OPTIMIZER: GradientDescentParametersSchema,
        L_BFGS_B_OPTIMIZER: LBFGSBParametersSchema,
        }


class OptimizerInfo(StrictMappingSchema):

    """Schema specifying the behavior of the multistarted optimizers in the optimal_learning library.

    .. Note:: This schema does not provide default values for its fields. These defaults
      ***DO EXIST***; see :mod:`moe.optimal_learning.python.constant`. However the defaults are
      dependent on external factors (like whether we're computing EI, log marginal, etc.) and
      are not known statically.

      See :meth:`moe.views.optimizable_gp_pretty_view.OptimizableGpPrettyView.get_params_from_request`
      for an example of how this schema is used.

    .. Note:: The field :attr:`optimizer_parameters` is ***NOT VALIDATED***. Users of this
      schema are responsible for passing its contents through the appropriate schema using
      the :const:`moe.views.schemas.OPTIMIZER_TYPES_TO_SCHEMA_CLASSES` dict provided above.

    TODO(GH-303): Try schema bindings as a way to automate setting validators and missing values.

    **Optional fields**

    :ivar optimizer_type: (*str*) the optimization type from :const:`moe.optimal_learning.python.constant.OPTIMIZER_TYPES` (default: GRADIENT_DESCENT_OPTIMIZER)
    :ivar num_multistarts: (*int > 0*) number of locations from which to start optimization runs
    :ivar num_random_samples: (*int >= 0*) number of random search points to use if multistart optimization fails
    :ivar optimizer_parameters: (*dict*) a dict corresponding the the parameters of the optimization method

    """

    optimizer_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(OPTIMIZER_TYPES),
            missing=None,
            )
    num_multistarts = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=None,
            )
    num_random_samples = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=0),
            missing=None,
            )
    # TODO(GH-303): Use schema binding to set up missing/default and validation dynamically
    optimizer_parameters = colander.SchemaNode(
            colander.Mapping(unknown='preserve'),
            missing=None,
            )


class GpNextPointsRequest(StrictMappingSchema):

    """A request colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    .. Warning:: Requesting ``num_to_sample`` > ``num_sampled`` can lead to singular GP-variance
      matrices and other conditioning issues (e.g., the resulting points may be tightly
      clustered). We suggest bootstrapping with a few grid points, random search, etc.

    See additional notes in :class:`moe.views.schemas.CovarianceInfo`,
    :class:`moe.views.schemas.BoundedDomainInfo`, :class:`moe.views.schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: (*int*) number of next points to generate (default: 1)
    :ivar mc_iterations: (*int*) number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
    :ivar max_num_threads: (*int*) maximum number of threads to use in computation
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information
    :ivar optimizer_info: (:class:`moe.views.schemas.OptimizerInfo`) dict of optimizer information
    :ivar points_being_sampled: (*list of float64*) points in domain being sampled in concurrent experiments (default: [])

    **General Timing Results**

    Here are some "broad-strokes" timing results for EI optimization.
    These tests are not complete nor comprehensive; they're just a starting point.
    The tests were run on a Sandy Bridge 2.3 GHz quad-core CPU. Data was generated
    from a Gaussian Process prior. The optimization parameters were the default
    values (see :mod:`moe.optimal_learning.python.constant`) as of sha
    ``c19257049f16036e5e2823df87fbe0812720e291``.

    Below, ``N = num_sampled``, ``MC = num_mc_iterations``, and ``q = num_to_sample``

    .. Note:: constant liar, kriging, and EPI (with ``num_to_sample = 1``) all fall
      under the ``analytic EI`` name.

    .. Note:: EI optimization times can vary widely as some randomly generated test
      cases are very "easy." Thus we give rough ranges.

    =============  =======================
     Analytic EI
    --------------------------------------
      dim, N           Gradient Descent
    =============  =======================
      3, 20               0.3 - 0.6s
      3, 40               0.5 - 1.4s
      3, 120              0.8 - 2.9s
      6, 20               0.4 - 0.9s
      6, 40              1.25 - 1.8s
      6, 120              2.9 - 5.0s
    =============  =======================

    We expect this to scale as ``~ O(dim)`` and ``~ O(N^3)``. The ``O(N^3)`` only happens
    once per multistart. Per iteration there's an ``O(N^2)`` dependence but as you can
    see, the dependence on ``dim`` is stronger.

    =============  =======================
     MC EI (``N = 20``, ``dim = 3``)
    --------------------------------------
      q, MC           Gradient Descent
    =============  =======================
      2, 10k                50 - 80s
      4, 10k              120 - 180s
      8, 10k              400 - 580s
      2, 40k              230 - 480s
      4, 40k              600 - 700s
    =============  =======================

    We expect this to scale as ``~ O(q^2)`` and ``~ O(MC)``. Scaling with ``dim`` and ``N``
    should be similar to the analytic case.

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
            "optimizer_info": {
                "optimizer_type": "gradient_descent_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimizer_parameters": {
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
            validator=colander.Range(min=1, max=MAX_ALLOWED_NUM_THREADS),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = BoundedDomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )
    optimizer_info = OptimizerInfo(
            missing=OptimizerInfo().deserialize({}),
            )
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )


class GpNextPointsConstantLiarRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.schemas.GpNextPointsRequest` with a lie value, for use with :class:`moe.views.rest.gp_next_points_constant_liar.GpNextPointsConstantLiar`.

    See :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.constant_liar_expected_improvement_optimization` for more info.

    .. Warning:: Setting :attr:`lie_value` ``< best_so_far`` (``= min(points_sampled_value)``)
      will lead to poor results. The resulting ``points_to_sample`` will be tightly clustered.
      Such results are generally of low value and may cause singular GP-variance matrices too.

    .. Warning:: Setting :attr:`lie_noise_variance` to 0 may cause singular GP covariance
      matrices when paired with large ``num_to_sample`` (for the same reason given in
      :class:`moe.views.schemas.GpHistoricalInfo`).

      Setting large :attr:`lie_noise_variance` may cause the output ``points_to_sample``
      to cluster--if one heuristic estimate is good and has large noise, MOE will want to
      increase resample that location to increase certainty.

    See additional notes/warnings in :class:`moe.views.schemas.GpNextPointsRequest`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: (*int*) number of next points to generate (default: 1)
    :ivar lie_method: (*str*) name from `CONSTANT_LIAR_METHODS` representing the liar method to use (default: 'constant_liar_min')
    :ivar lie_value: (*float64*) the 'lie' the Constant Liar heuristic will use (default: None). If `lie_value` is not None the algorithm will use this value instead of one calculated using `lie_method`.
    :ivar lie_noise_variance: (*float64 >= 0.0*) the noise variance of the 'lie' value (default: 0.0)
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information
    :ivar optimiaztion_info: (:class:`moe.views.schemas.OptimizerInfo`) dict of optimization information

    **General Timing Results**

    See the ``Analytic EI`` table in :class:`moe.views.schemas.GpNextPointsRequest` for
    rough timing numbers.

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "lie_value": 0.0,
            "lie_noise_variance": 1e-12,
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
            missing=DEFAULT_CONSTANT_LIAR_METHOD,
            validator=colander.OneOf(CONSTANT_LIAR_METHODS),
            )
    lie_value = colander.SchemaNode(
            colander.Float(),
            missing=None,
            )
    lie_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.schemas.GpNextPointsRequest` with kriging parameters, for use with :class:`moe.views.rest.gp_next_points_kriging.GpNextPointsKriging`.

    See :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` for more info.

    .. Warning:: Setting :attr:`kriging_noise_variance` to 0 may cause singular GP covariance
      matrices when paired with large ``num_to_sample`` (for the same reason given in
      :class:`moe.views.schemas.GpHistoricalInfo`).

      Setting large :attr:`kriging_noise_variance` may cause the output ``points_to_sample``
      to cluster--if one heuristic estimate is good and has large noise, MOE will want to
      increase resample that location to increase certainty.

    See additional notes/warnings in :class:`moe.views.schemas.GpNextPointsRequest`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: number of next points to generate (default: 1)
    :ivar std_deviation_coef: (*float64*) amount of GP-variance to add to each Kriging estimate, see
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` (default: 0.0)
    :ivar kriging_noise_variance: (*float64 >= 0.0*) noise variance for each Kriging estimate, see
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` (default: 0.0)
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information
    :ivar optimiaztion_info: (:class:`moe.views.schemas.OptimizerInfo`) dict of optimization information

    **General Timing Results**

    See the ``Analytic EI`` table in :class:`moe.views.schemas.GpNextPointsRequest` for
    rough timing numbers.

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
            missing=DEFAULT_KRIGING_STD_DEVIATION_COEF,
            )
    kriging_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_KRIGING_NOISE_VARIANCE,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsStatus(StrictMappingSchema):

    """A status schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar expected_improvement: (*float64 >= 0.0*) EI evaluated at ``points_to_sample`` (:class:`moe.views.schemas.ListOfExpectedImprovements`)
    :ivar optimizer_success: (*dict*) Whether or not the optimizer converged to an optimal set of ``points_to_sample``

    """

    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )
    optimizer_success = colander.SchemaNode(
        colander.Mapping(unknown='preserve'),
        default={'found_update': False},
    )


class GpNextPointsResponse(StrictMappingSchema):

    """A response colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar points_to_sample: (*list of list of float64*) points in the domain to sample next (:class:`moe.views.schemas.ListOfPointsInDomain`)
    :ivar status: (:class:`moe.views.schemas.GpNextPointsStatus`) dict indicating final EI value and
      optimization status messages (e.g., success)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint": "gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "status": {
                "expected_improvement": "0.443478498868",
                "optimizer_success": {
                    'gradient_descent_tensor_product_domain_found_update': True,
                    },
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    status = GpNextPointsStatus()


class GpHyperOptRequest(StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` request colander schema.

    .. Note:: Particularly when the amount of historical data is low, the log likelihood
      may grow toward extreme hyperparameter values (i.e., toward 0 or infinity). Select
      reasonable domain bounds. For example, in a driving distance parameter, the scale
      of feet is irrelevant, as is the scale of 1000s of miles.

    .. Note:: MOE's default optimization parameters were tuned for hyperparameter values roughly in [0.01, 100].
      Venturing too far out of this range means the defaults may perform poorly.

    See additional notes in :class:`moe.views.schemas.CovarianceInfo`,
    :class:`moe.views.schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) object of historical data
    :ivar domain_info: (:class:`moe.views.schemas.DomainInfo`) dict of domain information for the GP
    :ivar hyperparameter_domain_info: (:class:`moe.views.schemas.BoundedDomainInfo`) dict of domain information for the hyperparameter optimization

    **Optional fields**

    :ivar max_num_threads: (*int*) maximum number of threads to use in computation
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information, used as a starting point for optimization
    :ivar optimizer_info: (:class:`moe.views.schemas.OptimizerInfo`) dict of optimizer information

    **General Timing Results**

    Here are some "broad-strokes" timing results for hyperparameter optimization.
    These tests are not complete nor comprehensive; they're just a starting point.
    The tests were run on a Sandy Bridge 2.3 GHz quad-core CPU. Data was generated
    from a Gaussian Process prior. The optimization parameters were the default
    values (see :mod:`moe.optimal_learning.python.constant`) as of sha
    ``c19257049f16036e5e2823df87fbe0812720e291``.

    Below, ``N = num_sampled``.

    ======== ===================== ========================
    Scaling with dim (N = 40)
    -------------------------------------------------------
      dim     Gradient Descent             Newton
    ======== ===================== ========================
      3           85s                      3.6s
      6           80s                      7.2s
      12         108s                     19.5s
    ======== ===================== ========================

    GD scales ``~ O(dim)`` and Newton ``~ O(dim^2)`` although these dim values
    are not large enough to show the asymptotic behavior.

    ======== ===================== ========================
    Scaling with N (dim = 3)
    -------------------------------------------------------
      N       Gradient Descent             Newton
    ======== ===================== ========================
      20        14s                       0.72s
      40        85s                        3.6s
      120       2100s                       60s
    ======== ===================== ========================

    Both methods scale as ``~ O(N^3)`` which is clearly shown here.

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
            "optimizer_info": {
                "optimizer_type": "newton_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimizer_parameters": {
                    "gamma": 1.2,
                    ...
                    },
                },
            "log_likelihood_info": "log_marginal_likelihood"
        }

    """

    max_num_threads = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1, max=MAX_ALLOWED_NUM_THREADS),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )
    hyperparameter_domain_info = BoundedDomainInfo()
    optimizer_info = OptimizerInfo(
            missing=OptimizerInfo().deserialize({}),
            )
    log_likelihood_info = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(LIKELIHOOD_TYPES),
            missing=LOG_MARGINAL_LIKELIHOOD,
            )


class GpHyperOptStatus(StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` status schema.

    **Output fields**

    :ivar log_likelihood: (*float64*) The log likelihood at the new hyperparameters
    :ivar grad_log_likelihood: (*list of float64*) The gradient of the log likelihood at the new hyperparameters
    :ivar optimizer_success: (*dict*) Whether or not the optimizer converged to an optimal set of hyperparameters

    """

    log_likelihood = colander.SchemaNode(colander.Float())
    grad_log_likelihood = ListOfFloats()
    optimizer_success = colander.SchemaNode(
        colander.Mapping(unknown='preserve'),
        default={'found_update': False},
    )


class GpHyperOptResponse(StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information
    :ivar status: (:class:`moe.views.schemas.GpHyperOptStatus`) dict indicating final log likelihood value/gradient and
      optimization status messages (e.g., success)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_hyper_opt",
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": ["0.88", "1.24"],
                },
            "status": {
                "log_likelihood": "-37.3279872",
                "grad_log_likelihood: ["-3.8897e-12", "1.32789789e-11"],
                "optimizer_success": {
                        'newton_found_update': True,
                    },
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    covariance_info = CovarianceInfo()
    status = GpHyperOptStatus()


class GpMeanVarRequest(StrictMappingSchema):

    """A request colander schema for the views in :mod:`moe.views.rest.gp_mean_var`.

    .. Note:: Requesting ``points_to_evaluate`` that are close to each or close to existing
      ``points_sampled`` may result in a [numerically] singular GP-variance matrix.

    See additional notes in :class:`moe.views.schemas.CovarianceInfo`,
    :class:`moe.views.schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar points_to_evaluate: (*list of list of float64*) points in domain to calculate the Gaussian Process (GP) mean and covariance at (:class:`moe.views.schemas.ListOfPointsInDomain`)
    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) object of historical data

    **Optional fields**

    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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

    points_to_evaluate = ListOfPointsInDomain()
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )


class GpEndpointResponse(StrictMappingSchema):

    """A base schema for the endpoint name.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_mean_var",
        }

    """

    endpoint = colander.SchemaNode(colander.String())


class GpMeanMixinResponse(StrictMappingSchema):

    """A mixin response colander schema for the mean of a gaussian process.

    **Output fields**

    :ivar mean: (*list of float64*) the means of the GP at ``points_to_evaluate`` (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "mean": ["0.0873832198661","0.0130505261903","0.174755506336"],
        }

    """

    mean = ListOfFloats()


class GpVarMixinResponse(StrictMappingSchema):

    """A mixin response colander schema for the [co]variance of a gaussian process.

    **Output fields**

    :ivar var: (:class:`moe.views.schemas.MatrixOfFloats`) matrix of covariance of the GP at ``points_to_evaluate``

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


class GpVarDiagMixinResponse(StrictMappingSchema):

    """A mixin response colander schema for the variance of a gaussian process.

    **Output fields**

    :ivar var: (*list of float64*) variances of the GP at ``points_to_evaluate``; i.e., diagonal of the ``var`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": ["0.228910114429","0.996177332647","0.228910114429"],
        }

    """

    var = ListOfFloats()


class GpMeanResponse(GpEndpointResponse, GpMeanMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpVarResponse(GpEndpointResponse, GpVarMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpVarView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpVarDiagResponse(GpEndpointResponse, GpVarDiagMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpVarDiagView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpMeanVarResponse(GpMeanResponse, GpVarMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanVarView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpMeanVarDiagResponse(GpMeanResponse, GpVarDiagMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanVarDiagView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpEiRequest(StrictMappingSchema):

    """A :class:`moe.views.rest.gp_ei.GpEiView` request colander schema.

    .. Note:: Requesting ``points_to_evaluate`` and ``points_being_sampled`` that are close
      to each or close to existing ``points_sampled`` may result in a [numerically] singular
      GP-variance matrix (which is required by EI).

    See additional notes in :class:`moe.views.schemas.CovarianceInfo`,
    :class:`moe.views.schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar points_to_evaluate: (*list of list of float64*) points in domain to calculate Expected Improvement (EI) at (:class:`moe.views.schemas.ListOfPointsInDomain`)
    :ivar gp_historical_info: (:class:`moe.views.schemas.GpHistoricalInfo`) object of historical data

    **Optional fields**

    :ivar points_being_sampled: (*list of list of float64*) points in domain being sampled in concurrent experiments (default: []) (:class:`moe.views.schemas.ListOfPointsInDomain`)
    :ivar mc_iterations: (*int*) number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
    :ivar max_num_threads: (*int*) maximum number of threads to use in computation (default: 1)
    :ivar covariance_info: (:class:`moe.views.schemas.CovarianceInfo`) dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
        }

    """

    points_to_evaluate = ListOfPointsInDomain()
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            )
    max_num_threads = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1, max=MAX_ALLOWED_NUM_THREADS),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )


class GpEiResponse(StrictMappingSchema):

    """A :class:`moe.views.rest.gp_ei.GpEiView` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar expected_improvement: (*list of float64*) calculated expected improvements (:class:`moe.views.schemas.ListOfExpectedImprovements`)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "expected_improvement":["0.197246898375","0.443163755117","0.155819546878"]
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = ListOfExpectedImprovements()
