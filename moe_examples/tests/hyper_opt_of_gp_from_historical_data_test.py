# -*- coding: utf-8 -*-
"""Integration test for hyper_opt_of_gp_from_historical_data MOE example."""
import testify as T

from moe.optimal_learning.python.constant import TEST_OPTIMIZER_MULTISTARTS, TEST_OPTIMIZER_NUM_RANDOM_SAMPLES, TEST_GRADIENT_DESCENT_PARAMETERS, GRADIENT_DESCENT_OPTIMIZER

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.hyper_opt_of_gp_from_historical_data import run_example


class HyperOptOfGpFromHistoricalDataTest(MoeExampleTestCase):

    """Test the hyper_opt_of_gp_from_historical_data MOE example."""

    def test_example_runs(self):
        """Simple integration test for example."""
        run_example(
                verbose=False,
                testapp=self.testapp,
                optimizer_info={
                    'optimizer_type': GRADIENT_DESCENT_OPTIMIZER,
                    'num_multistarts': TEST_OPTIMIZER_MULTISTARTS,
                    'num_random_samples': TEST_OPTIMIZER_NUM_RANDOM_SAMPLES,
                    'optimizer_parameters': TEST_GRADIENT_DESCENT_PARAMETERS._asdict(),
                    })


if __name__ == "__main__":
    T.run()
