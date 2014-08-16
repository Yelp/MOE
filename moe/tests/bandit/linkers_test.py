# -*- coding: utf-8 -*-
"""Tests that linkers contain all possible types defined in constants."""
import testify as T

from moe.bandit.constant import BANDIT_ENDPOINTS, EPSILON_SUBTYPES, UCB_SUBTYPES
from moe.bandit.linkers import BANDIT_ENDPOINTS_TO_SUBTYPES, EPSILON_SUBTYPES_TO_BANDIT_METHODS, UCB_SUBTYPES_TO_BANDIT_METHODS


class LinkersTest(T.TestCase):

    """Tests that linkers contain all possible types defined in constants."""

    def test_bandit_links_have_all_bandit_endpoints(self):
        """Test each bandit endpoint is in a linker, and every linker key is a bandit endpoint."""
        T.assert_equal(
                set(BANDIT_ENDPOINTS),
                set(BANDIT_ENDPOINTS_TO_SUBTYPES.keys())
                )

    def test_epsilon_links_have_all_epsilon_subtypes(self):
        """Test each epsilon subtype is in a linker, and every linker key is an epsilon subtype."""
        T.assert_equal(
                set(EPSILON_SUBTYPES),
                set(EPSILON_SUBTYPES_TO_BANDIT_METHODS.keys())
                )

    def test_ucb_links_have_all_ucb_subtypes(self):
        """Test each UCB subtype is in a linker, and every linker key is a UCB subtype."""
        T.assert_equal(
                set(UCB_SUBTYPES),
                set(UCB_SUBTYPES_TO_BANDIT_METHODS.keys())
                )

if __name__ == "__main__":
    T.run()
