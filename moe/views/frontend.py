# -*- coding: utf-8 -*-
"""Frontend views for the MOE app."""
from pyramid.view import view_config

from moe.optimal_learning.python.constant import DEFAULT_GAUSSIAN_PROCESS_PARAMETERS, DEMO_GRADIENT_DESCENT_PARAMETERS, PRETTY_OPTIMIZER_MULTISTARTS


@view_config(route_name='home', renderer='moe:templates/index.mako')
def index_page(request):
    """The MOE index view.

    .. http:get:: /

    """
    return {
            'nav_active': 'home',
            }


@view_config(route_name='gp_plot', renderer='moe:templates/gp_plot.mako')
def gp_plot_page(request):
    """The MOE demo view.

    .. http:get:: /demo

    """
    return {
            'nav_active': 'demo',
            'default_gaussian_process_parameters': DEFAULT_GAUSSIAN_PROCESS_PARAMETERS,
            'default_ei_optimizer_parameters': DEMO_GRADIENT_DESCENT_PARAMETERS,
            'default_num_multistarts': PRETTY_OPTIMIZER_MULTISTARTS,
            }
