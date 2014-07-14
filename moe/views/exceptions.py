# -*- coding: utf-8 -*-
"""Views for handling Python exceptions.

Includes special views that catch the following:
    1. colander.Invalid

"""
import logging
import pprint

import colander

from pyramid.response import Response
from pyramid.view import view_config


@view_config(context=Exception)
def pyramid_error_view(exception, request):
    """Register a pyramid view to handle exceptions; i.e., log them and generate a Response.

    :param exception: exception to be handled
    :type exception: Exception (Python base exception type) (i.e., type of ``context``)
    :param request: the pyramid request that lead to the exception being raised.
    :type request: pyramid.request.Request
    :return: the pyramid response to be rendered
    :rtype: pyramid.response.Response

    """
    if "log" not in general_error.__dict__:
        general_error.log = logging.getLogger(__name__)

    # Log exception with traceback
    general_error.log.error(request)
    general_error.log.exception(exception)

    # Specialized handlers for certain exception types
    if issubclass(type(exception), colander.Invalid):
        return failed_colander_validation(exception, request)

    return general_error(exception, request)


def general_error(exception, request):
    """Catch any Python ``Exception``.

    :param exception: exception to be handled
    :type exception: Exception
    :param request: the pyramid request that lead to the exception being raised.
    :type request: pyramid.request.Request
    :return: the pyramid response to be rendered
    :rtype: pyramid.response.Response

    """
    status_int = 500
    body = '{0:d}: {1:s}\n{2:s}'.format(status_int, request.referrer, exception)
    response = Response(body=body, status_int=status_int)
    return response


def failed_colander_validation(exception, request):
    """Catch ``colander.Invalid`` and give an informative 500 response.

    :param exception: exception to be handled
    :type exception: colander.Invalid
    :param request: the pyramid request that lead to the exception being raised.
    :type request: pyramid.request.Request
    :return: the pyramid response to be rendered
    :rtype: pyramid.response.Response

    """
    status_int = 500
    body = '{0:d}: {1:s}\nFailed validation:\n{2:s}'.format(status_int, request.referrer, pprint.pformat(exception.asdict()))
    response = Response(body=body, status_int=status_int)
    return response
