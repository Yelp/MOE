# -*- coding: utf-8 -*-
"""A superclass to encapsulate getting optimizer parameters for views."""
from moe.optimal_learning.python.constant import OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS
from moe.views.constant import GP_NEXT_POINTS_EPI_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import OPTIMIZER_TYPES_TO_SCHEMA_CLASSES


class OptimizableGpPrettyView(GpPrettyView):

    """A superclass to encapsulate getting optimizer parameters for views."""

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.

        We explicitly pull out the ``optimizer_type`` and use it to deserialize and validate the parameters.
        This is necessary because we cannot dynamically assign the parameters schema within colander for every ``optimizer_type``.

        :returns: A deserialized self.request_schema object

        """
        # First we get the standard params (not including optimizer info)
        params = super(OptimizableGpPrettyView, self).get_params_from_request()
        optimizer_type = params['optimizer_info']['optimizer_type']
        # Find the schma class that corresponds to the ``optimizer_type`` of the request
        schema_class = OPTIMIZER_TYPES_TO_SCHEMA_CLASSES[optimizer_type]()

        # Create a default dictionary for the optimizer parameters
        # TODO(eliu): HACKY! Not sure how else to get around special casing EI
        optimizer_parameters_lookup = (optimizer_type, self._route_name)
        if self._route_name == GP_NEXT_POINTS_EPI_ROUTE_NAME:
            num_to_sample = params.get('num_to_sample')
            if num_to_sample == 1:
                optimizer_parameters_lookup += ('analytic', )
            else:
                optimizer_parameters_lookup += ('monte_carlo', )

        optimizer_parameters_dict = dict(OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS[optimizer_parameters_lookup].optimizer_parameters._asdict())

        # Override the defaults with information that may be in the optimizer parameters
        for param, val in self.request.json_body.get('optimizer_info', {}).get('optimizer_parameters', {}).iteritems():
            optimizer_parameters_dict[param] = val

        # Deserialize and validate the parameters
        validated_optimizer_parameters = schema_class.deserialize(optimizer_parameters_dict)

        # Put the now validated parameters back into the params dictionary to be consumed by the view
        params['optimizer_info']['optimizer_parameters'] = validated_optimizer_parameters
        return params
