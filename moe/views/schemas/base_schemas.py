# -*- coding: utf-8 -*-
"""Base level schemas for the response/request schemas of each MOE REST endpoint.

.. Warning:: Outputs of colander schema serialization/deserialization should be treated as
  READ-ONLY. It appears that "missing=" and "default=" value are weak-copied (by reference).
  Thus changing missing/default fields in the output dict can modify the schema!

"""
import colander

from moe.optimal_learning.python.constant import GRADIENT_DESCENT_OPTIMIZER, TENSOR_PRODUCT_DOMAIN_TYPE, SQUARE_EXPONENTIAL_COVARIANCE_TYPE, NULL_OPTIMIZER, NEWTON_OPTIMIZER, DOMAIN_TYPES, OPTIMIZER_TYPES, COVARIANCE_TYPES


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

    :ivar point: (:class:`moe.views.schemas.base_schemas.ListOfFloats`) The point sampled (in the domain of the function)
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

    All required fields from :class:`~moe.views.schemas.base_schemas.DomainInfo`

    :ivar domain_bounds: (*list of list of float64*) the bounds of the domain of type :class:`moe.views.schemas.base_schemas.Domain`

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

    .. Warning:: If the points are too close together (relative to the length scales in :class:`moe.views.schemas.base_schemas.CovarianceInfo`)
      with simultaneously very low or zero noise, the condition number of the GPP's covariance matrix can be very large. The
      matrix may even become numerically singular.

      In such cases, check for (nearly) duplicates points and be mindful of large length scales.

    .. Warning:: 0 ``noise_variance`` in the input historical data may lead to [numerically] singular covariance matrices. This
      becomes more likely as ``num_sampled`` increases. Noise caps the condition number at roughly ``1.0 / min(noise)``, so
      adding artificial noise (e.g., ``1.0e-12``) can aid with conditioning issues.

      MOE does not do this for you automatically since 0 noise may be extremely important for some users.

    .. Note:: MOE performs best if the input ``points_sampled_value`` are 0 mean.

    **Required fields**

    :ivar points_sampled: (*list of PointsSampled*) The :class:`moe.views.schemas.base_schemas.PointsSampled` (point, value, noise) that make up
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
#: optimizer schemas, e.g., :class:`moe.views.schemas.base_schemas.NewtonParametersSchema`.
OPTIMIZER_TYPES_TO_SCHEMA_CLASSES = {
        NULL_OPTIMIZER: NullParametersSchema,
        NEWTON_OPTIMIZER: NewtonParametersSchema,
        GRADIENT_DESCENT_OPTIMIZER: GradientDescentParametersSchema,
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
      the :const:`moe.views.schemas.base_schemas.OPTIMIZER_TYPES_TO_SCHEMA_CLASSES` dict provided above.

    .. Note:: specifying :attr:`num_multistarts` = 0 or :attr:`num_random_samples` = 0 in the POST
      request will set them to their default values internally.

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
