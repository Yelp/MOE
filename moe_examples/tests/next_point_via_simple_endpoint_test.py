# -*- coding: utf-8 -*-
"""Integration test for next_point_via_simple_endpoint MOE example."""
import testify as T

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.next_point_via_simple_endpoint import run_example

class NextPointsViaSimpleEndpointTest(MoeExampleTestCase):

    """Test the next_point_via_simple_endpoint MOE example."""

    def test_example_runs(self):
        """Simple integration test for example."""
        run_example(num_points_to_sample=1, testapp=self.testapp)

if __name__ == "__main__":
    T.run()

