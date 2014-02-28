# -*- coding: utf-8 -*-

import testify as T
import os

from tests.EPI.src.python.gaussian_process_test_case import GaussianProcessTestCase
from optimal_learning.EPI.src.python.models.plottable_optimal_gaussian_process import PlottableOptimalGaussianProcess
from optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points

class PlotMarginalLikelihoodTest(GaussianProcessTestCase):
	"""Tests optimal_learning/EPI/src/python/models/plottable_optimal_gaussian_process.py
	plot_marginal_likelihood_vs_alpha_and_length
	"""

	def _spin_up_a_test_GP(self):
		domain = [[0.0, 10.0]]
		# spin up a GP
		covariance_of_process = self._make_default_covariance_of_process(
				signal_variance=1.0,
				length=[2.0]
				)
		GP = self._make_default_gaussian_process(
				gaussian_process_class=PlottableOptimalGaussianProcess,
				domain=domain,
				default_sample_variance=0.01,
				covariance_of_process=covariance_of_process
				)

		# sample some random points
		points_to_sample = get_latin_hypercube_points(20, domain)
		self._sample_points_from_gaussian_process(
				GP,
				points_to_sample
				)

		return GP

	def test_GP_plot_figure_is_generated(self):
		"""This is where a docstring would go...
		"""
		domain = [[-10.0, 10.0]]
		# spin up a GP
		covariance_of_process = self._make_default_covariance_of_process(
				signal_variance=0.5,
				length=[2.0]
				)
		GP = self._make_default_gaussian_process(
				gaussian_process_class=PlottableOptimalGaussianProcess,
				domain=domain,
				covariance_of_process=covariance_of_process
				)

		# sample some random points
		points_to_sample = get_latin_hypercube_points(5, domain)
		self._sample_points_from_gaussian_process(
				GP,
				points_to_sample
				)

		# try to plot it
		GP.plot_gp_over_one_dim_range(
				resolution=200,
				save_figure=True,
				figure_path='static/img/tmp_GP_plot.png'
				)
		T.assert_equal(True, os.path.isfile('static/img/tmp_GP_plot.png'))

	def test_marginal_plot_figure_is_generated(self):
		"""This is where a docstring would go...
		"""
		GP = self._spin_up_a_test_GP()

		# try to plot it
		GP.plot_marginal_likelihood_vs_alpha_and_length(
				resolution=25,
				length_log_scale=False,
				gd_points = [[1.0], [2.0]],
				save_figure=True,
				figure_path='static/img/tmp_marginal_plot.png'
				)
		T.assert_equal(True, os.path.isfile('static/img/tmp_marginal_plot.png'))
