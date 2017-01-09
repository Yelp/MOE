# -*- coding: utf-8 -*-
"""Request/response schemas for ``bandit_epsilon`` endpoints."""
import colander

from moe.bandit.constant import DEFAULT_EPSILON_SUBTYPE, EPSILON_SUBTYPES
from moe.views.schemas import base_schemas
from moe.views.schemas.bandit_pretty_view import BanditHistoricalInfo


class BanditEpsilonRequest(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_epsilon` request colander schema.

    **Required fields**

    :ivar historical_info: (:class:`moe.views.schemas.bandit_pretty_view.BanditHistoricalInfo`) object of historical data describing arm performance

    **Optional fields**

    :ivar subtype: (*str*) subtype of the epsilon bandit algorithm (default: greedy)
    :ivar hyperparameter_info: (:class:`~moe.views.schemas.bandit_pretty_view.BanditEpsilonFirstHyperparameterInfo` or :class:`~moe.views.schemas.bandit_pretty_view.BanditEpsilonGreedyHyperparameterInfo`) dict of hyperparameter information

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
            missing=DEFAULT_EPSILON_SUBTYPE,
            )
    historical_info = BanditHistoricalInfo()
    hyperparameter_info = colander.SchemaNode(
            colander.Mapping(unknown='preserve'),
            missing={},
            )
