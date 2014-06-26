# -*- coding: utf-8 -*-
"""Base level schemas for the response/request schemas of each MOE REST endpoint."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_NEWTON_PARAMETERS, DEFAULT_GRADIENT_DESCENT_PARAMETERS, GRADIENT_DESCENT_OPTIMIZER, DEFAULT_OPTIMIZATION_MULTISTARTS, DEFAULT_OPTIMIZATION_NUM_RANDOM_SAMPLES, TENSOR_PRODUCT_DOMAIN_TYPE, SQUARE_EXPONENTIAL_COVARIANCE_TYPE, NULL_OPTIMIZER, NEWTON_OPTIMIZER
from moe.optimal_learning.python.linkers import DOMAIN_TYPES_TO_DOMAIN_LINKS, OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS, COVARIANCE_TYPES_TO_CLASSES


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
            raise colander.Invalid(node, msg='Value = {:f} must be positive and finite.'.format(cstruct))


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
            validator=colander.Range(min=0),
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

        :domain_type: the type of domain to use in ``moe.optimal_learning.python.python_version.domain.DOMAIN_TYPES_TO_DOMAIN_LINKS`` (default: TENSOR_PRODUCT_DOMAIN_TYPE)

    """

    domain_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(DOMAIN_TYPES_TO_DOMAIN_LINKS),
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

        :covariance_type: a covariance type in ``moe.optimal_learning.python.python_version.covariance.COVARIANCE_TYPES_TO_CLASSES``
        :hyperparameters: the hyperparameters corresponding to the given covariance_type

    """

    covariance_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(COVARIANCE_TYPES_TO_CLASSES),
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

        :optimization_type: a string defining the optimization type from `moe.optimal_learning.python.cpp_wrappers.optimization.OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS` (default: GRADIENT_DESCENT_OPTIMIZER)
        :optimization_parameters: a dict corresponding the the parameters of the optimization method

    """

    optimization_type = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS),
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
