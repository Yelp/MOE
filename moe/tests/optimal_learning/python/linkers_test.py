# -*- coding: utf-8 -*-
"""Tests that linkers contain all possible types defined in constants."""
from builtins import object
from moe.optimal_learning.python.constant import COVARIANCE_TYPES, DOMAIN_TYPES, OPTIMIZER_TYPES, LIKELIHOOD_TYPES
from moe.optimal_learning.python.linkers import COVARIANCE_TYPES_TO_CLASSES, DOMAIN_TYPES_TO_DOMAIN_LINKS, OPTIMIZER_TYPES_TO_OPTIMIZER_METHODS, LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS


class TestLinkers(object):

    """Tests that linkers contain all possible types defined in constants."""

    def test_covariance_links_have_all_covariance_types(self):
        """Test each covariance type is in a linker, and every linker key is a covariance type."""
        assert set(COVARIANCE_TYPES) == set(COVARIANCE_TYPES_TO_CLASSES.keys())

    def test_domain_links_have_all_domain_types(self):
        """Test each domain type is in a linker, and every linker is a domain type."""
        assert set(DOMAIN_TYPES) == set(DOMAIN_TYPES_TO_DOMAIN_LINKS.keys())

    def test_optimization_links_have_all_optimizer_types(self):
        """Test each optimizer type is in a linker, and every linker key is a optimizer type."""
        assert set(OPTIMIZER_TYPES) == set(OPTIMIZER_TYPES_TO_OPTIMIZER_METHODS.keys())

    def test_likelihood_links_have_all_likelihood_types(self):
        """Test each likelihood type is in a linker, and every linker key is a likelihood type."""
        assert set(LIKELIHOOD_TYPES) == set(LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS.keys())
