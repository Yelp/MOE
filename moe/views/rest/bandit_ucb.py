# -*- coding: utf-8 -*-
"""Classes for ``bandit_ucb`` endpoints.

Includes:

    1. pretty and backend views

"""
from pyramid.view import view_config

from moe.bandit.constant import UCB_SUBTYPE_1
from moe.bandit.linkers import UCB_SUBTYPES_TO_BANDIT_METHODS
from moe.views.bandit_pretty_view import BanditPrettyView
from moe.views.constant import BANDIT_UCB_ROUTE_NAME, BANDIT_UCB_PRETTY_ROUTE_NAME
from moe.views.pretty_view import PRETTY_RENDERER
from moe.views.schemas.bandit_pretty_view import BanditResponse
from moe.views.schemas.rest.bandit_ucb import BanditUCBRequest
from moe.views.utils import _make_bandit_historical_info_from_params


class BanditUCBView(BanditPrettyView):

    """Views for bandit_ucb endpoints."""

    _route_name = BANDIT_UCB_ROUTE_NAME
    _pretty_route_name = BANDIT_UCB_PRETTY_ROUTE_NAME

    request_schema = BanditUCBRequest()
    response_schema = BanditResponse()

    _pretty_default_request = {
            "subtype": UCB_SUBTYPE_1,
            "historical_info": BanditPrettyView._pretty_default_historical_info,
            }

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /bandit/ucb/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def bandit_ucb_view(self):
        """Endpoint for bandit_epsilon POST requests.

        .. http:post:: /bandit/ucb

           Predict the optimal arm from a set of arms, given historical data.

           :input: :class:`moe.views.schemas.rest.bandit_ucb.BanditUCBRequest`
           :output: :class:`moe.views.schemas.bandit_pretty_view.BanditResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        subtype = params.get('subtype')
        historical_info = _make_bandit_historical_info_from_params(params)

        bandit_class = UCB_SUBTYPES_TO_BANDIT_METHODS[subtype].bandit_class(historical_info=historical_info)
        arms_to_allocations = bandit_class.allocate_arms()

        return self.form_response({
                'endpoint': self._route_name,
                'arm_allocations': arms_to_allocations,
                'winner': bandit_class.choose_arm(arms_to_allocations),
                })
