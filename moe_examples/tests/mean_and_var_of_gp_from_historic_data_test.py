# -*- coding: utf-8 -*-
"""Integration test for mean_and_var_of_gp_from_historic_data MOE example."""
import testify as T

from moe_examples.tests.moe_example_test_case import MoeExampleTestCase
from moe_examples.mean_and_var_of_gp_from_historic_data import run_example


class MeanAndVarOfGpFromHistoricDataTest(MoeExampleTestCase):

    """Test the mean_and_var_of_gp_from_historic_data MOE example."""

    def test_example_runs(self):
        """Simple integration test for example."""
        run_example(
                verbose=False,
                testapp=self.testapp,
                rest_port=1337,
                )


if __name__ == "__main__":
    T.run()
