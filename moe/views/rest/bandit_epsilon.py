# -*- coding: utf-8 -*-
"""Classes for ``bandit_epsilon`` endpoints.

Includes:

    1. pretty and backend views

"""
from pyramid.view import view_config

from moe.bandit.constant import DEFAULT_EPSILON, EPSILON_SUBTYPE_GREEDY
from moe.bandit.linkers import EPSILON_SUBTYPES_TO_EPSILON_METHODS
from moe.views.bandit_pretty_view import BanditPrettyView
from moe.views.constant import BANDIT_EPSILON_ROUTE_NAME, BANDIT_EPSILON_PRETTY_ROUTE_NAME
from moe.views.pretty_view import PRETTY_RENDERER
from moe.views.schemas.rest.bandit_epsilon import BanditEpsilonRequest, BanditEpsilonResponse
from moe.views.utils import _make_bandit_historical_info_from_params


class BanditEpsilonView(BanditPrettyView):

    """Views for bandit_epsilon endpoints."""

    _route_name = BANDIT_EPSILON_ROUTE_NAME
    _pretty_route_name = BANDIT_EPSILON_PRETTY_ROUTE_NAME

    request_schema = BanditEpsilonRequest()
    response_schema = BanditEpsilonResponse()

    _pretty_default_request = {
            "subtype": EPSILON_SUBTYPE_GREEDY,
            "historical_info": BanditPrettyView._pretty_default_historical_info,
            "hyperparameter_info": {"epsilon": DEFAULT_EPSILON},
            }

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /bandit/epsilon/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def bandit_epsilon_view(self):
        """Endpoint for bandit_epsilon POST requests.

        .. http:post:: /bandit/epsilon

           Predict the optimal arm from a set of arms, given historical data.

           :input: :class:`moe.views.schemas.rest.bandit_epsilon.BanditEpsilonRequest`
           :output: :class:`moe.views.schemas.rest.bandit_epsilon.BanditEpsilonResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        subtype = params.get('subtype')
        historical_info = _make_bandit_historical_info_from_params(params)
        epsilon = params.get('hyperparameter_info').get('epsilon')

        bandit_class = EPSILON_SUBTYPES_TO_EPSILON_METHODS[subtype].bandit_class(historical_info=historical_info, epsilon=epsilon)

        return self.form_response({
                'endpoint': self._route_name,
                'arm_allocations': bandit_class.allocate_arms(),
                'winner': bandit_class.choose_arm(),
                })
