# -*- coding: utf-8 -*-
"""Request/response schemas for ``bandit_ucb`` endpoints."""
import colander

from moe.bandit.constant import DEFAULT_UCB_SUBTYPE, UCB_SUBTYPES
from moe.views.schemas import base_schemas
from moe.views.schemas.bandit_pretty_view import BanditHistoricalInfo


class BanditUCBRequest(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_ucb` request colander schema.

    **Required fields**

    :ivar historical_info: (:class:`moe.views.schemas.bandit_pretty_view.BanditHistoricalInfo`) object of historical data describing arm performance

    **Optional fields**

    :ivar subtype: (*str*) subtype of the UCB bandit algorithm (default: 1)

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
            "subtype": "UCB1-tuned",
            "historical_info": {
                "arms_sampled": {
                    "arm1": {"win": 20, "loss": 5, "total": 25, "variance": 0.1},
                    "arm2": {"win": 20, "loss": 10, "total": 30, "variance": 0.2},
                    "arm3": {"win": 0, "loss": 0, "total": 0},
                    },
                },
        }

    """

    subtype = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(UCB_SUBTYPES),
            missing=DEFAULT_UCB_SUBTYPE,
            )
    historical_info = BanditHistoricalInfo()
