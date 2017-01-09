# -*- coding: utf-8 -*-
"""Base pyramid app for MOE."""
from pyramid.config import Configurator

from moe.resources import Root
from moe.views.constant import ALL_MOE_ROUTES


#: Following the versioning system at http://semver.org/
#: See also docs/contributing.rst, section ``Versioning``
#: MAJOR: incremented for incompatible API changes
MAJOR = 0
#: MINOR: incremented for adding functionality in a backwards-compatible manner
MINOR = 2
#: PATCH: incremented for backward-compatible bug fixes and minor capability improvements
PATCH = 2
#: Latest release version of MOE
__version__ = "{0:d}.{1:d}.{2:d}".format(MAJOR, MINOR, PATCH)


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

    app = config.make_wsgi_app()

    # Message to the user
    print """
    Congratulations! MOE is now running.

    You can access the web interface at: http://localhost:6543

    Repo: https://github.com/Yelp/MOE
    Docs: http://yelp.github.io/MOE

    Note: If you installed MOE within a docker container you may need to specify the IP address of the VM instead of localhost.
    In OSX and Windows this is the startup information when you run boot2docker, or can be set in $DOCKER_HOST.
    """

    return app
