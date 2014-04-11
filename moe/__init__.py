from pyramid.config import Configurator
from pyramid.events import subscriber
from pyramid.events import NewRequest

from moe.resources import Root

def main(global_config, **settings):
    """ This function returns a WSGI application.
    """
    config = Configurator(settings=settings, root_factory=Root)
    config.add_static_view('static', 'moe:static')

    # Routes
    config.add_route('home', '/')
    config.add_route('gp_ei', '/gp/ei')
    config.add_route('gp_plot', '/gp/plot')
    config.add_route('gp_ei_pretty', '/gp/ei/pretty')
    config.add_route('gp_mean_var', '/gp/mean_var')
    config.add_route('gp_mean_var_pretty', '/gp/mean_var/pretty')
    config.add_route('gp_next_points_epi', '/gp/next_points/epi')
    config.add_route('gp_next_points_epi_pretty', '/gp/next_points/epi/pretty')

    # MongoDB
    if settings['use_mongo'] == 'true':
        import pymongo
        def add_mongo_db(event):
            settings = event.request.registry.settings
            url = settings['mongodb.url']
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
                'moe.optimal_learning.EPI.src.python.lib.cuda_linkers',
                'moe.optimal_learning.EPI.src.python.lib.plotter',
                'moe.optimal_learning.EPI.src.python.models.plottable_optimal_gaussian_process',
                'moe.tests',
                ],
            )
    return config.make_wsgi_app()
