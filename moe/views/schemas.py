"""Base level schemas for the response/request schemas of each MOE REST endpoint."""
import colander

from moe.optimal_learning.EPI.src.python.constant import default_gaussian_process_parameters, default_ei_optimization_parameters


class ListOfFloats(colander.SequenceSchema):

    """Colander list of floats."""

    float_in_list = colander.SchemaNode(colander.Float())


class SinglePoint(colander.MappingSchema):

    """A point object.

    Contains:
        point - ListOfFloats
        value - float
        value_var - float >= 0.0

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


class DomainCoordinate(colander.SequenceSchema):

    """A single domain interval."""

    domain_coordinate = colander.SchemaNode(colander.Float())


class Domain(colander.SequenceSchema):

    """A list of domain interval DomainCoordinate objects."""

    domain_coordinates = DomainCoordinate()


class GpInfo(colander.MappingSchema):

    """The Gaussian Process info needed for every request.

    Contains:
        points_sampled - PointsSampled
        domain - Domain
        length_scale - ListOfFloats
        signal_variance - float

    """

    points_sampled = PointsSampled()
    domain = Domain()
    length_scale = ListOfFloats(
            missing=default_gaussian_process_parameters.length_scale,
            )
    signal_variance = colander.SchemaNode(
            colander.Float(),
            missing=default_gaussian_process_parameters.signal_variance,
            )


class EiOptimizationParameters(colander.MappingSchema):

    """Optimization parameters."""

    num_multistarts = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.num_multistarts,
            validator=colander.Range(min=1),
            )
    gd_iterations = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.gd_iterations,
            validator=colander.Range(min=10),
            )
    max_num_restarts = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.max_num_restarts,
            validator=colander.Range(min=1),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.gamma,
            validator=colander.Range(min=0.0, max=1.0),
            )
    pre_mult = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.pre_mult,
            validator=colander.Range(min=0.0),
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.mc_iterations,
            validator=colander.Range(min=100),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.max_relative_change,
            validator=colander.Range(min=0.0, max=1.0),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.tolerance,
            validator=colander.Range(min=1.0e-15, max=1.0e-4),
            )


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
