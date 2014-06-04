# -*- coding: utf-8 -*-
"""A superclass to encapsulate getting optimization parameters for views."""
from moe.optimal_learning.python.constant import OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES


class OptimizableGpPrettyView(GpPrettyView):

    """A superclass to encapsulate getting optimization parameters for views."""

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.

        We explicitly pull out the ``optimization_type`` and use it to deserialize and validate the parameters.
        This is necessary because we cannot dynamically assign the parameters schema within colander for every ``optimization_type``.

        :returns: A deserialized self.request_schema object

        """
        # Deserialize the basic request schema
        params = self.request_schema.deserialize(self.request.json_body)
        optimization_type = params['optimization_info']['optimization_type']
        # Find the schma class that corresponds to the ``optimization_type`` of the request
        schema_class = OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES[optimization_type]()
        # Create a default dictionary for the optimization parameters
        optimization_parameters_dict = dict(OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS[optimization_type]._asdict())
        # Override the defaults with information that may be in the optimization parameters
        for param, val in self.request.json_body.get('optimization_info', {}).get('optimization_parameters', {}).iteritems():
            optimization_parameters_dict[param] = val

        # Deserialize and validate the parameters
        validated_optimization_parameters = schema_class.deserialize(optimization_parameters_dict)

        # Put the now validated parameters back into the params dictionary to be consumed by the view
        params['optimization_info']['optimization_parameters'] = validated_optimization_parameters
        return params
