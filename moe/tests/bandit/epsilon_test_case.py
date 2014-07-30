# -*- coding: utf-8 -*-
"""Base test case class for bandit epsilon tests; includes different epsilon values to test."""
from moe.tests.bandit.bandit_test_case import BanditTestCase


class EpsilonTestCase(BanditTestCase):

    """Base test case for the bandit library.

    Test different epsilon values.

    """

    epsilons_to_test = [0.0, 0.5, 1.0]
