# -*- coding: utf-8 -*-
"""A superclass to encapsulate getting optimizer parameters for views."""
import copy

from moe.optimal_learning.python.constant import OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS, ENDPOINT_TO_DEFAULT_OPTIMIZER_TYPE
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas.base_schemas import OptimizerInfo, OPTIMIZER_TYPES_TO_SCHEMA_CLASSES


class OptimizableGpPrettyView(GpPrettyView):

    """A superclass to encapsulate getting optimizer parameters for views."""

    def _get_default_optimizer_type(self, params):
        """Get the optimizer type associated with this REST endpoint.

        :param params: a (partially) deserialized REST request with everything except possibly
          ``params['optimizer_info']``
        :type params: dict
        :return: optimizer type to use, one of :const:`moe.optimal_learning.python.constant.OPTIMIZER_TYPES`
        :rtype: str

        """
        return ENDPOINT_TO_DEFAULT_OPTIMIZER_TYPE[self._route_name]

    def _get_default_optimizer_params(self, params):
        """Get the default optimizer parameters associated with the desired ``optimizer_type`` and REST endpoint.

        :param params: a (partially) deserialized REST request with everything except possibly
          ``params['optimizer_info']``
        :type params: dict
        :return: default multistart and optimizer parameters to use with this REST request
        :rtype: :class:`moe.optimal_learning.python.constant.DefaultOptimizerInfoTuple`

        """
        optimizer_type = params['optimizer_info']['optimizer_type']
        optimizer_parameters_lookup = (optimizer_type, self._route_name)
        return OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS[optimizer_parameters_lookup]

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.

        We explicitly pull out the ``optimizer_type`` and use it to deserialize and validate
        the other parameters (num_multistarts, num_random_samples, optimizer_parameters).

        This is necessary because we have different default optimizer parameters for
        different combinations of ``optimizer_type``, endpoint, and other features.
        See :const:`moe.optimal_learning.python.constants.OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS`

        :returns: A deserialized self.request_schema object
        :rtype: dict

        """
        # First we get the standard params (not including optimizer info)
        params = super(OptimizableGpPrettyView, self).get_params_from_request()

        # colander deserialized results are READ-ONLY. We will potentially be overwriting
        # fields of ``params['optimizer_info']``, so we need to copy it first.
        params['optimizer_info'] = copy.deepcopy(params['optimizer_info'])

        # Set optimizer_type to default value if the user did not provide a value
        if params['optimizer_info']['optimizer_type'] is None:
            params['optimizer_info']['optimizer_type'] = self._get_default_optimizer_type(params)

        default_optimizer_parameters = self._get_default_optimizer_params(params)

        # Set num_multistarts to default value if the user did not provide a value
        if params['optimizer_info']['num_multistarts'] is None:
            params['optimizer_info']['num_multistarts'] = default_optimizer_parameters.num_multistarts

        # Set num_random_samples to default value if the user did not provide a value
        if params['optimizer_info']['num_random_samples'] is None:
            params['optimizer_info']['num_random_samples'] = default_optimizer_parameters.num_random_samples

        # Override the defaults with information that may be in the optimizer parameters
        optimizer_parameters_dict = default_optimizer_parameters.optimizer_parameters._asdict()
        if params['optimizer_info']['optimizer_parameters']:
            for param, val in params['optimizer_info']['optimizer_parameters'].iteritems():
                optimizer_parameters_dict[param] = val

        # Find the schema class that corresponds to the ``optimizer_type`` of the request
        # TODO(GH-303): Until this ticket is complete (see schemas.OptimizerInfo),
        # optimizer_parameters has *not been validated yet*, so we need to validate manually.
        schema_class = OPTIMIZER_TYPES_TO_SCHEMA_CLASSES[params['optimizer_info']['optimizer_type']]()

        # Deserialize and validate the parameters
        validated_optimizer_parameters = schema_class.deserialize(optimizer_parameters_dict)

        # Put the now validated parameters back into the params dictionary to be consumed by the view
        params['optimizer_info']['optimizer_parameters'] = validated_optimizer_parameters

        # We may have filled in missing values; re-validate these values with deserialize()
        # and write the result into optimizer_info.
        params['optimizer_info'] = OptimizerInfo().deserialize(params['optimizer_info'])
        return params
