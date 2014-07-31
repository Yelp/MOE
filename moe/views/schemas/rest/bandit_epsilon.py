# -*- coding: utf-8 -*-
"""Request/response schemas for ``bandit_epsilon`` endpoints."""
import colander

from moe.bandit.constant import EPSILON_SUBTYPE_GREEDY, EPSILON_SUBTYPES
from moe.views.schemas import base_schemas
from moe.views.schemas.bandit_pretty_view import ArmAllocations, BanditEpsilonHyperparameterInfo, BanditHistoricalInfo


class BanditEpsilonRequest(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_epsilon` request colander schema.

    **Required fields**

    :ivar historical_info: (:class:`moe.views.schemas.bandit_pretty_view.BanditHistoricalInfo`) object of historical data describing arm performance

    **Optional fields**

    :ivar subtype: (*str*) subtype of the epsilon bandit algorithm (default: greedy)
    :ivar hyperparameter_info: (:class:`moe.views.schemas.bandit_pretty_view.BanditEpsilonHyperparameterInfo`) dict of hyperparameter information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "historical_info": {
                "arms_sampled": {
                    "arm1": {"win": 20, "loss": 5, "total": 25},
                    "arm2": {"win": 20, "loss": 10, "total": 30},
                    "arm3": {"win": 0, "loss": 0, "total": 0},
                    },
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "subtype": "greedy",
            "historical_info": {
                "arms_sampled": {
                    "arm1": {"win": 20, "loss": 5, "total": 25},
                    "arm2": {"win": 20, "loss": 10, "total": 30},
                    "arm3": {"win": 0, "loss": 0, "total": 0},
                    },
                },
            "hyperparameter_info": {
                "epsilon": 0.05,
                },
        }

    """

    subtype = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(EPSILON_SUBTYPES),
            missing=EPSILON_SUBTYPE_GREEDY,
            )
    historical_info = BanditHistoricalInfo()
    hyperparameter_info = BanditEpsilonHyperparameterInfo()


class BanditEpsilonResponse(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_epsilon` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar arms: (:class:`moe.views.schemas.bandit_pretty_view.ArmAllocations`) a dictionary of (arm name, allocaiton) key-value pairs
    :ivar winner: (*str*) winning arm name

    **Example Response**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "endpoint":"bandit_epsilon",
            "arm_allocations": {
                "arm1": 0.95,
                "arm2": 0.025,
                "arm3": 0.025,
                }
            "winner": "arm1",
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    arm_allocations = ArmAllocations()
    winner = colander.SchemaNode(colander.String())
