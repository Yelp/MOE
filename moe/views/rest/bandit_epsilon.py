# -*- coding: utf-8 -*-
"""Classes for ``bandit_epsilon`` endpoints.

Includes:

    1. pretty and backend views

"""
import copy

from pyramid.view import view_config

from moe.bandit.constant import DEFAULT_EPSILON, DEFAULT_EPSILON_SUBTYPE
from moe.bandit.linkers import EPSILON_SUBTYPES_TO_BANDIT_METHODS
from moe.views.bandit_pretty_view import BanditPrettyView
from moe.views.constant import BANDIT_EPSILON_ROUTE_NAME, BANDIT_EPSILON_PRETTY_ROUTE_NAME
from moe.views.pretty_view import PRETTY_RENDERER
from moe.views.schemas.bandit_pretty_view import BanditResponse, BANDIT_EPSILON_SUBTYPES_TO_HYPERPARAMETER_INFO_SCHEMA_CLASSES
from moe.views.schemas.rest.bandit_epsilon import BanditEpsilonRequest
from moe.views.utils import _make_bandit_historical_info_from_params


class BanditEpsilonView(BanditPrettyView):

    """Views for bandit_epsilon endpoints."""

    _route_name = BANDIT_EPSILON_ROUTE_NAME
    _pretty_route_name = BANDIT_EPSILON_PRETTY_ROUTE_NAME

    request_schema = BanditEpsilonRequest()
    response_schema = BanditResponse()

    _pretty_default_request = {
            "subtype": DEFAULT_EPSILON_SUBTYPE,
            "historical_info": BanditPrettyView._pretty_default_historical_info,
            "hyperparameter_info": {"epsilon": DEFAULT_EPSILON},
            }

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.

        We explicitly pull out the ``hyparparameter_info`` and use it to deserialize and validate
        the other parameters (epsilon, total_samples).

        This is necessary because we have different hyperparameters for
        different subtypes.

        :returns: A deserialized self.request_schema object
        :rtype: dict

        """
        # First we get the standard params (not including historical info)
        params = super(BanditEpsilonView, self).get_params_from_request()

        # colander deserialized results are READ-ONLY. We will potentially be overwriting
        # fields of ``params['hyperparameter_info']``, so we need to copy it first.
        params['hyperparameter_info'] = copy.deepcopy(params['hyperparameter_info'])

        # Find the schema class that corresponds to the ``subtype`` of the request
        # hyperparameter_info has *not been validated yet*, so we need to validate manually.
        schema_class = BANDIT_EPSILON_SUBTYPES_TO_HYPERPARAMETER_INFO_SCHEMA_CLASSES[params['subtype']]()

        # Deserialize and validate the parameters
        validated_hyperparameter_info = schema_class.deserialize(params['hyperparameter_info'])

        # Put the now validated hyperparameter info back into the params dictionary to be consumed by the view
        params['hyperparameter_info'] = validated_hyperparameter_info

        return params

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
           :output: :class:`moe.views.schemas.bandit_pretty_view.BanditResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        subtype = params.get('subtype')
        historical_info = _make_bandit_historical_info_from_params(params)

        bandit_class = EPSILON_SUBTYPES_TO_BANDIT_METHODS[subtype].bandit_class(historical_info=historical_info, **params.get('hyperparameter_info'))
        arms_to_allocations = bandit_class.allocate_arms()

        return self.form_response({
                'endpoint': self._route_name,
                'arm_allocations': arms_to_allocations,
                'winner': bandit_class.choose_arm(arms_to_allocations),
                })
