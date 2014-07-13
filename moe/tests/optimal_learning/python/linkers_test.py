# -*- coding: utf-8 -*-
"""Tests that linkers contain all possible types defined in constants."""
import testify as T

from moe.optimal_learning.python.constant import COVARIANCE_TYPES, DOMAIN_TYPES, OPTIMIZATION_TYPES, LIKELIHOOD_TYPES
from moe.optimal_learning.python.linkers import COVARIANCE_TYPES_TO_CLASSES, DOMAIN_TYPES_TO_DOMAIN_LINKS, OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS, LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS


class LinkersTest(T.TestCase):

    """Tests that linkers contain all possible types defined in constants."""

    def test_covariance_links_have_all_covariance_types(self):
        """Test each covariance type is in a linker, and every linker key is a covariance type."""
        T.assert_equal(
                set(COVARIANCE_TYPES),
                set(COVARIANCE_TYPES_TO_CLASSES.keys())
                )

    def test_domain_links_have_all_domain_types(self):
        """Test each domain type is in a linker, and every linker is a domain type."""
        T.assert_equal(
                set(DOMAIN_TYPES),
                set(DOMAIN_TYPES_TO_DOMAIN_LINKS.keys())
                )

    def test_optimization_links_have_all_optimization_types(self):
        """Test each optimization type is in a linker, and every linker key is a optimization type."""
        T.assert_equal(
                set(OPTIMIZATION_TYPES),
                set(OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS.keys())
                )

    def test_likelihood_links_have_all_likelihood_types(self):
        """Test each likelihood type is in a linker, and every linker key is a likelihood type."""
        T.assert_equal(
                set(LIKELIHOOD_TYPES),
                set(LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS.keys())
                )

if __name__ == "__main__":
    T.run()
