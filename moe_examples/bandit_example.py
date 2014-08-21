# -*- coding: utf-8 -*-
"""An example for accessing the all the types in moe/easy_interface/bandit_simple_endpoint.

The following function is used:

    1. :func:`moe.easy_interface.bandit_simple_endpoint.bandit`

The function requires some historical information to inform bandit.

We first sample [0,0] from the function and then generate and sample 5 optimal points from moe sequentially
We then update the hyperparameters of the GP (model selection)
This process is repeated until we have sampled 20 points in total
We then calculate the posterior mean and variance of the GP at several points
"""
from moe.bandit.constant import DEFAULT_BANDIT_HISTORICAL_INFO
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
    bandit_kwargs[BANDIT_BLA_ROUTE_NAME] = dict(kwargs.items() + bandit_bla_kwargs.items())

    if bandit_epsilon_kwargs is None:
        bandit_epsilon_kwargs = {}
    bandit_kwargs[BANDIT_EPSILON_ROUTE_NAME] = dict(kwargs.items() + bandit_epsilon_kwargs.items())

    if bandit_ucb_kwargs is None:
        bandit_ucb_kwargs = {}
    bandit_kwargs[BANDIT_UCB_ROUTE_NAME] = dict(kwargs.items() + bandit_ucb_kwargs.items())

    historical_info = _make_bandit_historical_info_from_params(DEFAULT_BANDIT_HISTORICAL_INFO, arm_type=BernoulliArm)

    for type in BANDIT_ROUTE_NAMES:
        if verbose:
            print "Running Bandit: {0:s}...".format(type)
        for subtype in BANDIT_ROUTE_NAMES_TO_SUBTYPES[type]:
            if verbose:
                print "Running subtype: {0:s}...".format(subtype)
            bandit_kwargs[type]['subtype'] = subtype
            arm_allocations = bandit(historical_info, type=type, testapp=testapp, **bandit_kwargs[type])
            if verbose:
                print "Arm allocations {0:s}".format(str(arm_allocations))


if __name__ == '__main__':
    run_example()
