# -*- coding: utf-8 -*-
"""Base schemas for creating SampledArm and allocations for bandit endpoints along with base request/response schema components."""
import colander

from moe.bandit.constant import DEFAULT_EPSILON, DEFAULT_TOTAL_SAMPLES, EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY
from moe.bandit.data_containers import SampleArm
from moe.views.schemas import base_schemas


class ArmAllocations(colander.MappingSchema):

    """Colander SingleArm Dictionary of (arm name, allocation) key-value pairs."""

    schema_type = colander.MappingSchema
    title = 'Arm Allocations'

    def __init__(self):
        """Allow any arm name to be a valid key."""
        super(ArmAllocations, self).__init__(colander.Mapping(unknown='preserve'))

    def validator(self, node, cstruct):
        """Raise an exception if the node value (cstruct) is not a valid dictionary of (arm name, allocation) key-value pairs.

        The total allocation must sums to 1.
        Each allocation is in range [0,1].

        :param node: the node being validated (usually self)
        :type node: colander.SchemaNode subclass instance
        :param cstruct: the value being validated
        :type cstruct: dictionary of (arm name, allocation) key-value pairs

        """
        total_allocation = 0.0
        for arm_name, allocation in cstruct.iteritems():
            total_allocation += allocation
            if not 0.0 <= allocation <= 1.0:
                raise colander.Invalid(node, msg='Allocation = {:f} must be in range [0,1].'.format(allocation))
        if total_allocation != 1.0:
            raise colander.Invalid(node, msg='Total Allocation = {:f} must be 1.0.'.format(total_allocation))


class ArmsSampled(colander.MappingSchema):

    """Colander SingleArm Dictionary of (arm name, SingleArm) key-value pairs."""

    schema_type = colander.MappingSchema
    title = 'Arms Sampled'

    def __init__(self):
        """Allow any arm name to be a valid key."""
        super(ArmsSampled, self).__init__(colander.Mapping(unknown='preserve'))

    def validator(self, node, cstruct):
        """Raise an exception if the node value (cstruct) is not a valid dictionary of (arm name, SingleArm) key-value pairs. Default value for loss is 0. Default value for variance of an arm is None.

        :param node: the node being validated (usually self)
        :type node: colander.SchemaNode subclass instance
        :param cstruct: the value being validated
        :type cstruct: dictionary of (arm name, SingleArm) key-value pairs

        """
        for arm_name, sample_arm in cstruct.iteritems():
            if 'loss' not in sample_arm:
                sample_arm['loss'] = 0
            if 'variance' not in sample_arm:
                sample_arm['variance'] = None
            if not (set(sample_arm.keys()) == set(['win', 'loss', 'total', 'variance'])):
                raise colander.Invalid(node, msg='Value = {:s} must be a valid SampleArm.'.format(sample_arm))
            SampleArm(sample_arm['win'], sample_arm['loss'], sample_arm['total'], sample_arm['variance'])


class BanditEpsilonFirstHyperparameterInfo(base_schemas.StrictMappingSchema):

    """The hyperparameter info needed for every  Bandit Epsilon-First request.

    **Required fields**

    :ivar epsilon: (*0.0 <= float64 <= 1.0*) epsilon value for epsilon-first bandit. This strategy pulls the optimal arm
      (best expected return) with if it is in exploitation phase (number sampled > epsilon * total_samples). Otherwise a random arm is pulled (exploration).
    :ivar total_samples: (*int >= 0*) total number of samples for epsilon-first bandit. total_samples is T from :doc:`bandit`.

    """

    epsilon = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0, max=1.0),
            missing=DEFAULT_EPSILON,
            )

    total_samples = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=0),
            missing=DEFAULT_TOTAL_SAMPLES,
            )


class BanditEpsilonGreedyHyperparameterInfo(base_schemas.StrictMappingSchema):

    """The hyperparameter info needed for every  Bandit Epsilon request.

    **Required fields**

    :ivar epsilon: (*0.0 <= float64 <= 1.0*) epsilon value for epsilon-greedy bandit. This strategy pulls the optimal arm
      (best expected return) with probability 1-epsilon. With probability epsilon a random arm is pulled.

    """

    epsilon = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0, max=1.0),
            missing=DEFAULT_EPSILON,
            )


#: Mapping from bandit epsilon subtypes (:const:`moe.bandit.constant.EPSILON_SUBTYPES`) to
#: hyperparameter info schemas, e.g., :class:`moe.views.schemas.bandit_pretty_view.BanditEpsilonFirstHyperparameterInfo`.
BANDIT_EPSILON_SUBTYPES_TO_HYPERPARAMETER_INFO_SCHEMA_CLASSES = {
        EPSILON_SUBTYPE_FIRST: BanditEpsilonFirstHyperparameterInfo,
        EPSILON_SUBTYPE_GREEDY: BanditEpsilonGreedyHyperparameterInfo,
        }


class BanditHistoricalInfo(base_schemas.StrictMappingSchema):

    """The Bandit historical info needed for every request.

    Contains:

        * arms_sampled - ArmsSampled

    """

    arms_sampled = ArmsSampled()


class BanditResponse(base_schemas.StrictMappingSchema):

    """A :mod:`moe.views.rest.bandit_epsilon` and :mod:`moe.views.rest.bandit_ucb` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar arms: (:class:`moe.views.schemas.bandit_pretty_view.ArmAllocations`) a dictionary of (arm name, allocation) key-value pairs
    :ivar winner: (*str*) winning arm name

    **Example Response**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "endpoint":"bandit_epsilon",
            "arm_allocations": {
                "arm1": 0.95,
                "arm2": 0.025,
                "arm3": 0.025,
                }
            "winner": "arm1",
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    arm_allocations = ArmAllocations()
    winner = colander.SchemaNode(colander.String())
