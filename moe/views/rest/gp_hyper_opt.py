# -*- coding: utf-8 -*-
"""Classes for gp_hyper_opt endpoints.

Includes:

    1. request and response schemas
    2. pretty and backend views

"""
from pyramid.view import view_config

from moe.optimal_learning.python.constant import OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS, ENDPOINT_TO_DEFAULT_OPTIMIZER_TYPE
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import multistart_hyperparameter_optimization
from moe.optimal_learning.python.linkers import LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS
from moe.optimal_learning.python.timing import timing_context
from moe.views.constant import GP_HYPER_OPT_ROUTE_NAME, GP_HYPER_OPT_PRETTY_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView, PRETTY_RENDERER
from moe.views.optimizable_gp_pretty_view import OptimizableGpPrettyView
from moe.views.schemas.rest.gp_hyper_opt import GpHyperOptRequest, GpHyperOptResponse
from moe.views.utils import _make_domain_from_params, _make_gp_from_params, _make_optimizer_parameters_from_params


MODEL_SELECTION_TIMING_LABEL = 'model selection time'


class GpHyperOptView(OptimizableGpPrettyView):

    """Views for gp_hyper_opt endpoints."""

    _route_name = GP_HYPER_OPT_ROUTE_NAME
    _pretty_route_name = GP_HYPER_OPT_PRETTY_ROUTE_NAME

    request_schema = GpHyperOptRequest()
    response_schema = GpHyperOptResponse()

    _pretty_default_request = {
            "gp_historical_info": GpPrettyView._pretty_default_gp_historical_info,
            "domain_info": {"dim": 1},
            "covariance_info": GpPrettyView._pretty_default_covariance_info,
            "hyperparameter_domain_info": {
                "dim": 2,
                "domain_bounds": [
                    {
                        "min": 0.1,
                        "max": 2.0,
                    },
                    {
                        "min": 0.1,
                        "max": 2.0,
                    },
                    ],
                },
            }

    def _get_default_optimizer_type(self, params):
        """Get the optimizer type associated with this REST endpoint and requested ``log_likelihood_type``.

        :param params: a (partially) deserialized REST request with everything except possibly
          ``params['optimizer_info']``
        :type params: dict
        :return: optimizer type to use, one of :const:`moe.optimal_learning.python.constant.OPTIMIZER_TYPES`
        :rtype: str

        """
        log_likelihood_type = params.get('log_likelihood_info')
        return ENDPOINT_TO_DEFAULT_OPTIMIZER_TYPE[(self._route_name, log_likelihood_type)]

    def _get_default_optimizer_params(self, params):
        """Get the default optimizer parameters associated with the desired ``optimizer_type``, REST endpoint, and ``log_likelihood_type``.

        :param params: a (partially) deserialized REST request with everything except possibly
          ``params['optimizer_info']``
        :type params: dict
        :return: default multistart and optimizer parameters to use with this REST request
        :rtype: :class:`moe.optimal_learning.python.constant.DefaultOptimizerInfoTuple`

        """
        optimizer_type = params['optimizer_info']['optimizer_type']
        log_likelihood_type = params.get('log_likelihood_info')
        optimizer_parameters_lookup = (optimizer_type, self._route_name, log_likelihood_type)
        return OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS[optimizer_parameters_lookup]

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/hyper_opt/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_hyper_opt_view(self):
        """Endpoint for gp_hyper_opt POST requests.

        .. http:post:: /gp/hyper_opt

           Calculates the optimal hyperparameters for a gaussian process, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_hyper_opt.GpHyperOptRequest`
           :output: :class:`moe.views.schemas.rest.gp_hyper_opt.GpHyperOptResponse`

           :status 201: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        max_num_threads = params.get('max_num_threads')
        hyperparameter_domain = _make_domain_from_params(params, domain_info_key='hyperparameter_domain_info')
        gaussian_process = _make_gp_from_params(params)
        covariance_of_process, historical_data = gaussian_process.get_core_data_copy()
        optimizer_class, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)
        log_likelihood_type = params.get('log_likelihood_info')

        log_likelihood_eval = LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS[log_likelihood_type].log_likelihood_class(
            covariance_of_process,
            historical_data,
        )

        log_likelihood_optimizer = optimizer_class(
            hyperparameter_domain,
            log_likelihood_eval,
            optimizer_parameters,
            num_random_samples=0,  # hyperopt doesn't use dumb search if optimization fails
        )

        hyperopt_status = {}
        with timing_context(MODEL_SELECTION_TIMING_LABEL):
            optimized_hyperparameters = multistart_hyperparameter_optimization(
                log_likelihood_optimizer,
                optimizer_parameters.num_multistarts,
                max_num_threads=max_num_threads,
                status=hyperopt_status,
            )

        covariance_of_process.hyperparameters = optimized_hyperparameters

        log_likelihood_eval.current_point = optimized_hyperparameters

        return self.form_response({
                'endpoint': self._route_name,
                'covariance_info': covariance_of_process.get_json_serializable_info(),
                'status': {
                    'log_likelihood': log_likelihood_eval.compute_log_likelihood(),
                    'grad_log_likelihood': log_likelihood_eval.compute_grad_log_likelihood().tolist(),
                    'optimizer_success': hyperopt_status,
                    },
                })
