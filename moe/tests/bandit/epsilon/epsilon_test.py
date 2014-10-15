# -*- coding: utf-8 -*-
"""Test epsilon bandit implementation (functions common to epsilon bandit).

Test functions in :class:`moe.bandit.epsilon.epsilon_interface.EpsilonInterface`

"""
import pytest

import logging

from moe.bandit.epsilon.epsilon_interface import EpsilonInterface
from moe.tests.bandit.epsilon.epsilon_test_case import EpsilonTestCase


class TestEpsilon(EpsilonTestCase):

    """Verify that different sample_arms return correct results."""

    @pytest.fixture()
    def disable_logging(self, request):
        """Disable logging (for the duration of this test case)."""
        logging.disable(logging.CRITICAL)

        def finalize():
            """Re-enable logging (so other test cases are unaffected)."""
            logging.disable(logging.NOTSET)
        request.addfinalizer(finalize)

    @pytest.mark.usefixtures("disable_logging")
    def test_empty_arm_invalid(self):
        """Test empty ``sample_arms`` causes an ValueError."""
        with pytest.raises(ValueError):
            EpsilonInterface.get_winning_arm_names({})

    def test_two_unsampled_arms(self):
        """Check that the two-unsampled-arms case always returns both arms as winning arms. This tests num_winning_arms == num_arms > 1."""
        assert EpsilonInterface.get_winning_arm_names(self.two_unsampled_arms_test_case.arms_sampled) == frozenset(["arm1", "arm2"])

    def test_three_arms_two_winners(self):
        """Check that the three-arms cases with two winners return the expected winning arms. This tests num_arms > num_winning_arms > 1."""
        assert EpsilonInterface.get_winning_arm_names(self.three_arms_two_winners_test_case.arms_sampled) == frozenset(["arm1", "arm2"])
