# -*- coding: utf-8 -*-
"""Integration test for bandit_example MOE example."""
from moe.bandit.constant import TEST_EPSILON

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.bandit_example import run_example


class TestBanditExample(MoeExampleTestCase):

    """Test the bandit_example MOE example."""

    def test_example_runs_with_non_default_kwargs(self):
        """Simple integration test for example with non default kwargs."""
        run_example(
                verbose=False,
                testapp=self.testapp,
                bandit_bla_kwargs={},
                bandit_epsilon_kwargs={
                    'hyperparameter_info': {
                        'epsilon': TEST_EPSILON,
                        }
                    },
                bandit_ucb_kwargs={},
                rest_port=1337,
                )
