# -*- coding: utf-8 -*-
"""Integration test for combined_example MOE example."""
from moe.optimal_learning.python.constant import TEST_OPTIMIZER_MULTISTARTS, TEST_OPTIMIZER_NUM_RANDOM_SAMPLES, TEST_GRADIENT_DESCENT_PARAMETERS

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.combined_example import run_example


class TestCombinedExample(MoeExampleTestCase):

    """Test the combined_example MOE example."""

    def test_example_runs_with_non_default_optimizer_kwargs(self):
        """Simple integration test for example with non default kwargs."""
        run_example(
                num_to_sample=1,
                verbose=False,
                testapp=self.testapp,
                gp_next_points_kwargs={
                    'optimizer_info': {
                        'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                        'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                        'optimizer_parameters': TEST_GRADIENT_DESCENT_PARAMETERS._asdict(),
                        }
                    },
                gp_hyper_opt_kwargs={
                    'optimizer_info': {
                        'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                        'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                        'optimizer_parameters': TEST_GRADIENT_DESCENT_PARAMETERS._asdict(),
                        }
                    },
                gp_mean_var_kwargs={},
                rest_port=1337,
                )
