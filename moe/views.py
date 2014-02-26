from pyramid.response import Response
from pyramid.view import view_config

@view_config(route_name='home', renderer='moe:templates/index.mako')
def index_page(request):
    return {
            'nav_active': 'home',
            }

@view_config(route_name='about', renderer='moe:templates/about.mako')
def about_page(request):
    return {
            'nav_active': 'about',
            }

@view_config(route_name='docs', renderer='moe:templates/docs.mako')
def docs_page(request):
    return {
            'nav_active': 'docs',
            }
