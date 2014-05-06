# -*- coding: utf-8 -*-
"""Base level schemas for the response/request schemas of each MOE REST endpoint."""
import colander

from moe.optimal_learning.python.constant import default_gaussian_process_parameters, default_ei_optimization_parameters, default_optimizer_type, default_num_random_samples, ALL_OPTIMIZERS


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


class DomainCoordinate(colander.SequenceSchema):

    """A single domain interval."""

    domain_coordinate = colander.SchemaNode(colander.Float())


class Domain(colander.SequenceSchema):

    """A list of domain interval DomainCoordinate objects."""

    domain_coordinates = DomainCoordinate()


class GpInfo(colander.MappingSchema):

    """The Gaussian Process info needed for every request.

    Contains:
        * points_sampled - PointsSampled
        * domain - Domain
        * length_scale - ListOfFloats
        * signal_variance - float

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

    """Optimization parameters.

    **Optional fields**

        :param optimizer_type: the type of optimizer to use
        :type optimizer_type: string in ['gradient_descent']
        :param num_random_samples: the number of random samples to try on top of the optimization method (failsafe)
        :type num_random_samples: int >= 0
        :param num_multistarts: number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
        :type num_multistarts: int > 0
        :param max_num_steps: maximum number of gradient descent iterations per restart (suggest: 200-1000)
        :type max_num_steps: int > 0
        :param max_num_restarts: maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 10-20)
        :type max_num_restarts: int > 0
        :param gamma: exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
        :type gamma: float64 > 1.0
        :param pre_mult: scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
        :type pre_mult: float64 > 0.0
        :param max_relative_change: max change allowed per GD iteration (as a relative fraction of current distance to wall)
               (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
        :type max_relative_change: float64 in [0, 1]
        :param tolerance: when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
               (e.g., at a boundary), stop.  (suggest: 1.0e-7)
        :type tolerance: float64 >= 0.0

    ***Example Request** (default values in moe/optimal_learning/python/constant)

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'optimizer_type': 'gradient_descent',
            'num_random_samples': 4000,
            'num_multistarts': 40,
            'gd_iterations': 1000,
            'max_num_restarts': 3,
            'gamma': 0.9,
            'pre_mult': 1.0,
            'mc_iterations': 100000,
            'max_relative_change': 1.0,
            'tolerance': 1.0e-7,
        }

    """

    optimizer_type = colander.SchemaNode(
            colander.String(),
            missing=default_optimizer_type,
            validator=colander.OneOf(ALL_OPTIMIZERS),
            )
    num_random_samples = colander.SchemaNode(
            colander.Int(),
            missing=default_num_random_samples,
            validator=colander.Range(min=0),
            )
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
