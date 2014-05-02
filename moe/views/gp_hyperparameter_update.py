# -*- coding: utf-8 -*-
"""Classes for gp_ei endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import colander
from pyramid.view import view_config


class GpEiRequest(colander.MappingSchema):

    """A gp_ei request colander schema.

    **Required fields**

        :gp_info: a moe.views.schemas.GpInfo object of historical data

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'gp_info': {
                'signal_variance': 1.0,
                'length_scale': [0.2],
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                'domain': [
                    [0, 1],
                    ]
                },
            },
        }

    """
