# -*- coding: utf-8 -*-
"""Base pyramid app for MOE."""
from pyramid.config import Configurator

from moe.resources import Root
from moe.views.constant import ALL_MOE_ROUTES


#: Latest release version of MOE
__version__ = '0.1.0'


def main(global_config, **settings):
    """Return a WSGI application."""
    config = Configurator(settings=settings, root_factory=Root)
    config.include('pyramid_mako')
    config.add_static_view('static', 'moe:static')

    # Routes
    config.add_route('home', '/')
    config.add_route('gp_plot', '/gp/plot')
    # MOE routes
    for moe_route in ALL_MOE_ROUTES:
        config.add_route(
                moe_route.route_name,
                moe_route.endpoint
                )

    config.scan(
            ignore=[
                'moe.optimal_learning.python.lib.cuda_linkers',
                'moe.tests',
                ],
            )
    return config.make_wsgi_app()
