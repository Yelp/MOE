# -*- coding: utf-8 -*-
"""An example for accessing the all the types in moe/easy_interface/bandit_simple_endpoint.

The following function is used:

    1. :func:`moe.easy_interface.bandit_simple_endpoint.bandit`

See :doc:`bandit` for more details on multi-armed bandits.
The function requires some historical information to inform bandit.

We compute arm allocations for all bandit type and subtypes with the simple example of Bernoulli arms.
"""
from __future__ import print_function
from builtins import str
from moe.bandit.data_containers import BernoulliArm
from moe.bandit.linkers import BANDIT_ROUTE_NAMES_TO_SUBTYPES
from moe.easy_interface.bandit_simple_endpoint import bandit
from moe.views.constant import BANDIT_BLA_ROUTE_NAME, BANDIT_EPSILON_ROUTE_NAME, BANDIT_UCB_ROUTE_NAME, BANDIT_ROUTE_NAMES
from moe.views.utils import _make_bandit_historical_info_from_params


def run_example(
        verbose=True,
        testapp=None,
        bandit_bla_kwargs=None,
        bandit_epsilon_kwargs=None,
        bandit_ucb_kwargs=None,
        **kwargs
):
    """Run the bandit example.

    :param verbose: Whether to print information to the screen [True]
    :type verbose: bool
    :param testapp: Whether to use a supplied test pyramid application or a rest server [None]
    :type testapp: Pyramid test application
    :param bandit_bla_kwargs: Optional kwargs to pass to bandit_bla endpoint
    :type bandit_bla_kwargs: dict
    :param bandit_epsilon_kwargs: Optional kwargs to pass to bandit_epsilon endpoint
    :type bandit_epsilon_kwargs: dict
    :param bandit_ucb_kwargs: Optional kwargs to pass to bandit_ucb endpoint
    :type bandit_ucb_kwargs: dict
    :param kwargs: Optional kwargs to pass to all endpoints
    :type kwargs: dict

    """
    # Set and combine all optional kwargs
    # Note that the more specific kwargs take precedence (and will override general kwargs)
    bandit_kwargs = {}
    if bandit_bla_kwargs is None:
        bandit_bla_kwargs = {}
    bandit_kwargs[BANDIT_BLA_ROUTE_NAME] = dict(list(kwargs.items()) + list(bandit_bla_kwargs.items()))

    if bandit_epsilon_kwargs is None:
        bandit_epsilon_kwargs = {}
    bandit_kwargs[BANDIT_EPSILON_ROUTE_NAME] = dict(list(kwargs.items()) + list(bandit_epsilon_kwargs.items()))

    if bandit_ucb_kwargs is None:
        bandit_ucb_kwargs = {}
    bandit_kwargs[BANDIT_UCB_ROUTE_NAME] = dict(list(kwargs.items()) + list(bandit_ucb_kwargs.items()))

    # A BernoulliArm has payoff 1 for a success and 0 for a failure.
    # See :class:`~moe.bandit.data_containers.BernoulliArm` for more details.
    historical_info = _make_bandit_historical_info_from_params(
            {
                "historical_info": {
                    "arms_sampled": {
                        "arm1": {"win": 20, "loss": 0, "total": 25},
                        "arm2": {"win": 20, "loss": 0, "total": 30},
                        "arm3": {"win": 0, "loss": 0, "total": 0},
                    }
                },
            },
            arm_type=BernoulliArm
            )

    # Run all multi-armed bandit strategies we have implemented. See :doc:`bandit` for more details on multi-armed bandits.
    # We have implemented 3 bandit types: BLA (Bayesian Learning Optimization), Epsilon, and UCB (Upper Confidence Bound).
    for type in BANDIT_ROUTE_NAMES:
        if verbose:
            print("Running Bandit: {0:s}...".format(type))
        # Each bandit type has different subtypes. If a user does not specify a subtype, we use the default subtype.
        # For example, the bandit type Epsilon has two subtypes: epsilon-first and epsilon-greedy.
        # See :class:`~moe.bandit.epsilon.epsilon_first.EpsilonFirst` and :class:`~moe.bandit.epsilon.epsilon_greedy.EpsilonGreedy` for more details.
        for subtype in BANDIT_ROUTE_NAMES_TO_SUBTYPES[type]:
            if verbose:
                print("Running subtype: {0:s}...".format(subtype))
            bandit_kwargs[type]['subtype'] = subtype
            # Compute and return arm allocations given the sample history of bandit arms.
            # For example, the allocations {arm1: 0.3, arm2: 0.7} means
            # if we have 10 arm pulls, we should pull arm1 3 times and arm2 7 times.
            # See :func:`moe.bandit.bandit_interface.BanditInterface.allocate_arms` for more details.
            arm_allocations = bandit(historical_info, type=type, testapp=testapp, **bandit_kwargs[type])
            if verbose:
                print("Arm allocations {0:s}".format(str(arm_allocations)))


if __name__ == '__main__':
    run_example()
