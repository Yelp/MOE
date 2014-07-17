# -*- coding: utf-8 -*-
"""Route names and endpoints for all MOE routes.

Regular GP REST routes:

    * ei - compute the Expected Improvement at a set of points
    * mean_var - compute the mean and variance of the gaussian process for a set of points

Next Points GP REST routes:

    * epi - compute the next points to sample using Expected Parallel Improvement
    * kriging - compute the next points to sample using Kriging Believer
    * constant_liar - compute the next points to sample using Constant Liar

New routes have the form:

    GP_<NAME>_ROUTE_NAME = 'gp_<name>'
    GP_<NAME>_ENDPOINT = '/gp/<name>'
    GP_<NAME>_MOE_ROUTE = MoeRoute(GP_<NAME>_ROUTE_NAME, GP_<NAME>_ENDPOINT)
    GP_<NAME>_PRETTY_ROUTE_NAME = 'gp_<name>_pretty'
    GP_<NAME>_PRETTY_ENDPOINT = '/gp/<name>/pretty'
    GP_<NAME>_PRETTY_MOE_ROUTE = MoeRoute(GP_<NAME>_PRETTY_ROUTE_NAME, GP_<NAME>_PRETTY_ENDPOINT)

New next_points routes have the form:

    GP_NEXT_POINTS_<NAME>_ROUTE_NAME = 'gp_next_points_<name>'
    GP_NEXT_POINTS_<NAME>_ENDPOINT = '/gp/next_points/<name>'
    GP_NEXT_POINTS_<NAME>_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_<NAME>_ROUTE_NAME, GP_NEXT_POINTS_<NAME>_ENDPOINT)
    GP_NEXT_POINTS_<NAME>_PRETTY_ROUTE_NAME = 'gp_next_points_<name>_pretty'
    GP_NEXT_POINTS_<NAME>_PRETTY_ENDPOINT = '/gp/next_points/<name>/pretty'
    GP_NEXT_POINTS_<NAME>_PRETTY_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_<NAME>_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_<NAME>_PRETTY_ENDPOINT)
    GP_NEXT_POINTS_<NAME>_OPTIMIZATION_METHOD_NAME = <method name from moe.optimal_learning.python.models.optimal_gaussian_process_linked_cpp.py>

"""
from collections import namedtuple


_BaseMoeRestLogLine = namedtuple('MoeLogLine', ['endpoint', 'type', 'content'])


class MoeRestLogLine(_BaseMoeRestLogLine):

    """The information logged for all MOE REST requests/responses.

    :ivar endpoint: The endpoint that was called
    :ivar type: Whether this is a ``'request'`` or ``'response'``
    :ivar content: The json of the request/response

    """

    __slots__ = ()

_BaseMoeRoute = namedtuple('MoeRoute', ['route_name', 'endpoint'])


class MoeRoute(_BaseMoeRoute):

    """Information for mapping a MOE ``route_name`` to its corresponding endpoint.

    :ivar route_name: The name of the route (ie ``'gp_ei'``)
    :ivar endpoint: The endpoint for the route (ie ``'/gp/ei'``)

    """

    __slots__ = ()

BANDIT_EPSILON_ROUTE_NAME = 'bandit_epsilon'
BANDIT_EPSILON_ENDPOINT = '/bandit/epsilon'
BANDIT_EPSILON_MOE_ROUTE = MoeRoute(BANDIT_EPSILON_ROUTE_NAME, BANDIT_EPSILON_ENDPOINT)
BANDIT_EPSILON_PRETTY_ROUTE_NAME = 'bandit_epsilon_pretty'
BANDIT_EPSILON_PRETTY_ENDPOINT = '/bandit/epsilon/pretty'
BANDIT_EPSILON_PRETTY_MOE_ROUTE = MoeRoute(BANDIT_EPSILON_PRETTY_ROUTE_NAME, BANDIT_EPSILON_PRETTY_ENDPOINT)

GP_EI_ROUTE_NAME = 'gp_ei'
GP_EI_ENDPOINT = '/gp/ei'
GP_EI_MOE_ROUTE = MoeRoute(GP_EI_ROUTE_NAME, GP_EI_ENDPOINT)
GP_EI_PRETTY_ROUTE_NAME = 'gp_ei_pretty'
GP_EI_PRETTY_ENDPOINT = '/gp/ei/pretty'
GP_EI_PRETTY_MOE_ROUTE = MoeRoute(GP_EI_PRETTY_ROUTE_NAME, GP_EI_PRETTY_ENDPOINT)

GP_MEAN_ROUTE_NAME = 'gp_mean'
GP_MEAN_ENDPOINT = '/gp/mean'
GP_MEAN_MOE_ROUTE = MoeRoute(GP_MEAN_ROUTE_NAME, GP_MEAN_ENDPOINT)
GP_MEAN_PRETTY_ROUTE_NAME = 'gp_mean_pretty'
GP_MEAN_PRETTY_ENDPOINT = '/gp/mean/pretty'
GP_MEAN_PRETTY_MOE_ROUTE = MoeRoute(GP_MEAN_PRETTY_ROUTE_NAME, GP_MEAN_PRETTY_ENDPOINT)

GP_VAR_ROUTE_NAME = 'gp_var'
GP_VAR_ENDPOINT = '/gp/var'
GP_VAR_MOE_ROUTE = MoeRoute(GP_VAR_ROUTE_NAME, GP_VAR_ENDPOINT)
GP_VAR_PRETTY_ROUTE_NAME = 'gp_var_pretty'
GP_VAR_PRETTY_ENDPOINT = '/gp/var/pretty'
GP_VAR_PRETTY_MOE_ROUTE = MoeRoute(GP_VAR_PRETTY_ROUTE_NAME, GP_VAR_PRETTY_ENDPOINT)

GP_VAR_DIAG_ROUTE_NAME = 'gp_var_diag'
GP_VAR_DIAG_ENDPOINT = '/gp/var/diag'
GP_VAR_DIAG_MOE_ROUTE = MoeRoute(GP_VAR_DIAG_ROUTE_NAME, GP_VAR_DIAG_ENDPOINT)
GP_VAR_DIAG_PRETTY_ROUTE_NAME = 'gp_var_diag_pretty'
GP_VAR_DIAG_PRETTY_ENDPOINT = '/gp/var/diag/pretty'
GP_VAR_DIAG_PRETTY_MOE_ROUTE = MoeRoute(GP_VAR_DIAG_PRETTY_ROUTE_NAME, GP_VAR_DIAG_PRETTY_ENDPOINT)

GP_MEAN_VAR_ROUTE_NAME = 'gp_mean_var'
GP_MEAN_VAR_ENDPOINT = '/gp/mean_var'
GP_MEAN_VAR_MOE_ROUTE = MoeRoute(GP_MEAN_VAR_ROUTE_NAME, GP_MEAN_VAR_ENDPOINT)
GP_MEAN_VAR_PRETTY_ROUTE_NAME = 'gp_mean_var_pretty'
GP_MEAN_VAR_PRETTY_ENDPOINT = '/gp/mean_var/pretty'
GP_MEAN_VAR_PRETTY_MOE_ROUTE = MoeRoute(GP_MEAN_VAR_PRETTY_ROUTE_NAME, GP_MEAN_VAR_PRETTY_ENDPOINT)

GP_MEAN_VAR_DIAG_ROUTE_NAME = 'gp_mean_var_diag'
GP_MEAN_VAR_DIAG_ENDPOINT = '/gp/mean_var/diag'
GP_MEAN_VAR_DIAG_MOE_ROUTE = MoeRoute(GP_MEAN_VAR_DIAG_ROUTE_NAME, GP_MEAN_VAR_DIAG_ENDPOINT)
GP_MEAN_VAR_DIAG_PRETTY_ROUTE_NAME = 'gp_mean_var_diag_pretty'
GP_MEAN_VAR_DIAG_PRETTY_ENDPOINT = '/gp/mean_var/diag/pretty'
GP_MEAN_VAR_DIAG_PRETTY_MOE_ROUTE = MoeRoute(GP_MEAN_VAR_DIAG_PRETTY_ROUTE_NAME, GP_MEAN_VAR_DIAG_PRETTY_ENDPOINT)

GP_NEXT_POINTS_EPI_ROUTE_NAME = 'gp_next_points_epi'
GP_NEXT_POINTS_EPI_ENDPOINT = '/gp/next_points/epi'
GP_NEXT_POINTS_EPI_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_NEXT_POINTS_EPI_ENDPOINT)
GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME = 'gp_next_points_epi_pretty'
GP_NEXT_POINTS_EPI_PRETTY_ENDPOINT = '/gp/next_points/epi/pretty'
GP_NEXT_POINTS_EPI_PRETTY_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_EPI_PRETTY_ENDPOINT)
GP_NEXT_POINTS_EPI_OPTIMIZATION_METHOD_NAME = 'multistart_expected_improvement_optimization'

GP_NEXT_POINTS_KRIGING_ROUTE_NAME = 'gp_next_points_kriging'
GP_NEXT_POINTS_KRIGING_ENDPOINT = '/gp/next_points/kriging'
GP_NEXT_POINTS_KRIGING_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_KRIGING_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_ENDPOINT)
GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME = 'gp_next_points_kriging_pretty'
GP_NEXT_POINTS_KRIGING_PRETTY_ENDPOINT = '/gp/next_points/kriging/pretty'
GP_NEXT_POINTS_KRIGING_PRETTY_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_PRETTY_ENDPOINT)
GP_NEXT_POINTS_KRIGING_OPTIMIZATION_METHOD_NAME = 'kriging_believer_expected_improvement_optimization'

GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME = 'gp_next_points_constant_liar'
GP_NEXT_POINTS_CONSTANT_LIAR_ENDPOINT = '/gp/next_points/constant_liar'
GP_NEXT_POINTS_CONSTANT_LIAR_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_ENDPOINT)
GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME = 'gp_next_points_constant_liar_pretty'
GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ENDPOINT = '/gp/next_points/constant_liar/pretty'
GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_MOE_ROUTE = MoeRoute(GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ENDPOINT)
GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZATION_METHOD_NAME = 'constant_liar_expected_improvement_optimization'

GP_HYPER_OPT_ROUTE_NAME = 'gp_hyper_opt'
GP_HYPER_OPT_ENDPOINT = '/gp/hyper_opt'
GP_HYPER_OPT_MOE_ROUTE = MoeRoute(GP_HYPER_OPT_ROUTE_NAME, GP_HYPER_OPT_ENDPOINT)
GP_HYPER_OPT_PRETTY_ROUTE_NAME = 'gp_hyper_opt_pretty'
GP_HYPER_OPT_PRETTY_ENDPOINT = '/gp/hyper_opt/pretty'
GP_HYPER_OPT_PRETTY_MOE_ROUTE = MoeRoute(GP_HYPER_OPT_PRETTY_ROUTE_NAME, GP_HYPER_OPT_PRETTY_ENDPOINT)

# These need to match method names in moe/optimal_learning/python/cpp_wrappers/expected_improvement.py
OPTIMIZATION_METHOD_NAMES = [
        GP_NEXT_POINTS_EPI_OPTIMIZATION_METHOD_NAME,
        GP_NEXT_POINTS_KRIGING_OPTIMIZATION_METHOD_NAME,
        GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZATION_METHOD_NAME,
        ]

ALL_REST_MOE_ROUTES = [
        BANDIT_EPSILON_MOE_ROUTE,
        GP_EI_MOE_ROUTE,
        GP_MEAN_MOE_ROUTE,
        GP_VAR_MOE_ROUTE,
        GP_VAR_DIAG_MOE_ROUTE,
        GP_MEAN_VAR_MOE_ROUTE,
        GP_MEAN_VAR_DIAG_MOE_ROUTE,
        GP_NEXT_POINTS_EPI_MOE_ROUTE,
        GP_NEXT_POINTS_KRIGING_MOE_ROUTE,
        GP_NEXT_POINTS_CONSTANT_LIAR_MOE_ROUTE,
        GP_HYPER_OPT_MOE_ROUTE,
        ]

ALL_NEXT_POINTS_MOE_ROUTES = [
        GP_NEXT_POINTS_EPI_MOE_ROUTE,
        GP_NEXT_POINTS_KRIGING_MOE_ROUTE,
        GP_NEXT_POINTS_CONSTANT_LIAR_MOE_ROUTE,
        ]

ALL_PRETTY_MOE_ROUTES = [
        BANDIT_EPSILON_PRETTY_MOE_ROUTE,
        GP_EI_PRETTY_MOE_ROUTE,
        GP_MEAN_PRETTY_MOE_ROUTE,
        GP_VAR_PRETTY_MOE_ROUTE,
        GP_VAR_DIAG_PRETTY_MOE_ROUTE,
        GP_MEAN_VAR_PRETTY_MOE_ROUTE,
        GP_MEAN_VAR_DIAG_PRETTY_MOE_ROUTE,
        GP_NEXT_POINTS_EPI_PRETTY_MOE_ROUTE,
        GP_NEXT_POINTS_KRIGING_PRETTY_MOE_ROUTE,
        GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_MOE_ROUTE,
        GP_HYPER_OPT_PRETTY_MOE_ROUTE,
        ]

ALL_MOE_ROUTES = []
ALL_MOE_ROUTES.extend(ALL_REST_MOE_ROUTES)
ALL_MOE_ROUTES.extend(ALL_PRETTY_MOE_ROUTES)

ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT = {}
for moe_route in ALL_REST_MOE_ROUTES:
    ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT[moe_route.route_name] = moe_route.endpoint
