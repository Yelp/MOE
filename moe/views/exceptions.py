# -*- coding: utf-8 -*-
"""Views for handling exceptions.

Includes views that catch the following:
    1. SingularMatrixError
    2. colander.Invalid

"""
import pprint

import colander

from pyramid.response import Response
from pyramid.view import view_config


class SingularMatrixError(Exception):

    """Throw this exception when you have constructed a singular matrix."""

    pass


@view_config(context=SingularMatrixError)
def linear_algebra_error(exception, request):
    """Catch SingularMatrixError and give an informative 500 response."""
    response = Response('SingularMatrixError, check points_sampled for identical points')
    response.status_int = 500
    print exception
    return response


@view_config(context=colander.Invalid)
def failed_validation(exception, request):
    """Catch colander.Invalid and give an informative 500 response."""
    response = Response('Failed validation:\n%s' % pprint.pformat(exception.asdict()))
    response.status_int = 500
    return response
