# -*- coding: utf-8 -*-
"""A superclass to encapsulate getting optimization parameters for views."""
from moe.optimal_learning.python.constant import OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES


class OptimizableGpPrettyView(GpPrettyView):

    """A superclass to encapsulate getting optimization parameters for views."""

    def get_params_from_request(self):
        """Return the deserialized parameters from the json_body of a request.

        :returns: A deserialized self.request_schema object

        """
        params = self.request_schema.deserialize(self.request.json_body)
        optimization_type = params['optimization_info']['optimization_type']
        schema_class = OPTIMIZATION_TYPES_TO_SCHEMA_CLASSES[optimization_type]()
        optimization_parameters_dict = dict(OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS[optimization_type]._asdict())
        for param, val in self.request.json_body.get('optimization_info', {}).get('optimization_parameters', {}).iteritems():
            optimization_parameters_dict[param] = val

        validated_optimization_parameters = schema_class.deserialize(optimization_parameters_dict)

        params['optimization_info']['optimization_parameters'] = validated_optimization_parameters
        return params
