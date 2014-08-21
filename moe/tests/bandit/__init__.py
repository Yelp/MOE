# -*- coding: utf-8 -*-
r"""Testing code for the (Python) bandit library.

Testing is done via the Testify package:
https://github.com/Yelp/Testify

This package includes:

* Test cases/test setup files
* Tests for bandit/epsilon
* Tests for bandit/ucb
* Tests for bandit/bla

This package includes:

* Test cases/test setup files
* Tests for classes and utils in :mod:`moe.bandit`

**Files in this package**

* :mod:`moe.tests.bandit.bandit_interface_test`: tests for :mod:`moe.bandit.interfaces.bandit_interface.BanditInterface`
* :mod:`moe.tests.bandit.bandit_test_case`: base test case for bandit tests with a simple integration test case
* :mod:`moe.tests.bandit.bla_test`: tests for :mod:`moe.bandit.bla.BLA`
* :mod:`moe.tests.bandit.epsilon_first_test`: tests for :mod:`moe.bandit.epsilon_greedy.EpsilonFirst`
* :mod:`moe.tests.bandit.epsilon_greedy_test`: tests for :mod:`moe.bandit.epsilon_greedy.EpsilonGreedy`
* :mod:`moe.tests.bandit.epsilon_test`: tests for :mod:`moe.bandit.epsilon_greedy.Epsilon`
* :mod:`moe.tests.bandit.epsilon_test_case`: test cases for classes under :mod:`moe.bandit.epsilon.Epsilon`
* :mod:`moe.tests.bandit.linkers_test`: tests for :mod:`moe.bandit.linkers`
* :mod:`moe.tests.bandit.ucb_test_case`: base test case for UCB tests
* :mod:`moe.tests.bandit.ucb1_test`: tests for :mod:`moe.bandit.ucb1.UCB1`
* :mod:`moe.tests.bandit.ucb1_tuned_test`: tests for :mod:`moe.bandit.ucb1_tuned.UCB1Tuned`
* :mod:`moe.tests.bandit.utils_test`: tests for :mod:`moe.bandit.utils`

"""
