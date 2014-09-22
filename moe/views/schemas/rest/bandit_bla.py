# -*- coding: utf-8 -*-
"""Request/response schemas for ``bandit_bla`` endpoints."""
import colander

from moe.bandit.constant import DEFAULT_BLA_SUBTYPE, BLA_SUBTYPES
from moe.views.schemas import base_schemas
from moe.views.schemas.bandit_pretty_view import BanditHistoricalInfo


class BanditBLARequest(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_bla` request colander schema.

    **Required fields**

    :ivar historical_info: (:class:`moe.views.schemas.bandit_pretty_view.BanditHistoricalInfo`) object of historical data describing arm performance

    **Optional fields**

    :ivar subtype: (*str*) subtype of the BLA bandit algorithm (default: BLA)

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "historical_info": {
                "arms_sampled": {
                    "arm1": {"win": 20, "total": 25},
                    "arm2": {"win": 20, "total": 30},
                    "arm3": {"win": 0, "total": 0},
                    },
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "subtype": "BLA",
            "historical_info": {
                "arms_sampled": {
                    "arm1": {"win": 20, "loss": 0, "total": 25},
                    "arm2": {"win": 20, "loss": 0, "total": 30},
                    "arm3": {"win": 0, "loss": 0, "total": 0},
                    },
                },
        }

    """

    subtype = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(BLA_SUBTYPES),
            missing=DEFAULT_BLA_SUBTYPE,
            )
    historical_info = BanditHistoricalInfo()
