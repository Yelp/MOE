# -*- coding: utf-8 -*-
"""Tests that linkers contain all possible types defined in constants."""
import testify as T

from moe.bandit.constant import EPSILON_SUBTYPES
from moe.bandit.linkers import EPSILON_SUBTYPES_TO_EPSILON_METHODS


class LinkersTest(T.TestCase):

    """Tests that linkers contain all possible types defined in constants."""

    def test_epsilon_links_have_all_epsilon_subtypes(self):
        """Test each epsilon subtype is in a linker, and every linker key is an epsilon subtype."""
        T.assert_equal(
                set(EPSILON_SUBTYPES),
                set(EPSILON_SUBTYPES_TO_EPSILON_METHODS.keys())
                )

if __name__ == "__main__":
    T.run()
