# -*- coding: utf-8 -*-
"""Base pyramid app for MOE."""
from pyramid.config import Configurator
from pyramid.events import NewRequest

from moe.resources import Root
from moe.views.constant import ALL_MOE_ROUTES


def main(global_config, **settings):
    """Return a WSGI application."""
    config = Configurator(settings=settings, root_factory=Root)
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

    # MongoDB
    if settings['use_mongo'] == 'true':
        import pymongo

        def add_mongo_db(event):
            settings = event.request.registry.settings
            db_name = settings['mongodb.db_name']
            db = settings['mongodb_conn'][db_name]
            event.request.db = db
        db_uri = settings['mongodb.url']
        db_port = int(settings['mongodb.port'])
        MongoDB = pymongo.Connection
        if 'pyramid_debugtoolbar' in set(settings.values()):
            class MongoDB(pymongo.Connection):
                def __html__(self):
                    return 'MongoDB: <b>{}></b>'.format(self)
        conn = MongoDB(
                db_uri,
                db_port,
                )
        config.registry.settings['mongodb_conn'] = conn
        config.add_subscriber(add_mongo_db, NewRequest)
    config.scan(
            ignore=[
                'moe.optimal_learning.python.lib.cuda_linkers',
                'moe.tests',
                ],
            )
    return config.make_wsgi_app()
