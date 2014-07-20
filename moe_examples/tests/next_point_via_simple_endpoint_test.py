# -*- coding: utf-8 -*-
"""Integration test for next_point_via_simple_endpoint MOE example."""
import testify as T

from moe.optimal_learning.python.constant import TEST_GRADIENT_DESCENT_PARAMETERS

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.next_point_via_simple_endpoint import run_example


class NextPointsViaSimpleEndpointTest(MoeExampleTestCase):

    """Test the next_point_via_simple_endpoint MOE example."""

    def test_example_runs(self):
        """Simple integration test for example."""
        run_example(
                num_points_to_sample=1,
                verbose=False,
                testapp=self.testapp,
                optimizer_info={
                    'optimizer_parameters': TEST_GRADIENT_DESCENT_PARAMETERS._asdict(),
                    })


if __name__ == "__main__":
    T.run()
