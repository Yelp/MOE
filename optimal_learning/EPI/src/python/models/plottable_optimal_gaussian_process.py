# -*- coding: utf-8 -*-

import numpy # for sci comp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt # for plotting

from optimal_learning.EPI.src.python.models.optimal_gaussian_process import OptimalGaussianProcess


class PlottableOptimalGaussianProcess(OptimalGaussianProcess):
	"""docstring!
	"""

	def plot_gp_over_one_dim_range(self, plotting_range=None, resolution=50, save_figure=False, figure_path=None):
		"""Plot the mean and variance of the gaussian process over some range
		"""
		if plotting_range is None:
			plotting_range = self.domain[0] # we can only plot in 1D

		points = numpy.arange(
				plotting_range[0],
				plotting_range[1],
				(plotting_range[1] - plotting_range[0])/resolution
				)

		mean_values = numpy.zeros(resolution)
		var_values = numpy.zeros(resolution)

		for i, point in enumerate(points):
			mean, var = self.get_mean_and_var_of_points([numpy.array([point])])
			mean_values[i] = mean[0]
			var_values[i] = var[0][0]

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# show sampled points with red x's
		sampled_x_points = numpy.zeros(len(self.points_sampled))
		sampled_x_vals = numpy.zeros(len(self.points_sampled))
		for i, sample in enumerate(self.points_sampled):
			sampled_x_points[i] = sample.point
			sampled_x_vals[i] = sample.value
		ax.plot(sampled_x_points, sampled_x_vals, 'rx')

		# plot mean and var accross the points
		ax.plot(points, mean_values, '--')
		ax.fill_between(points, mean_values - var_values, mean_values + var_values, facecolor='green', alpha=0.2)

		# set limits
		y_min = numpy.min([numpy.min(mean_values - var_values), numpy.min(sampled_x_vals)])
		y_max = numpy.max([numpy.max(mean_values + var_values), numpy.max(sampled_x_vals)])
		ax.set_ylim(y_min, y_max)
		ax.set_xlim(plotting_range[0], plotting_range[1])

		if save_figure:
			if not figure_path:
				return fig
			else:
				plt.savefig(figure_path, bbox_inches=0) # remove whitespace and save
		else:
			plt.show()

	def plot_marginal_likelihood_vs_alpha_and_length(self, alpha_range=[0.1, 2.0], length_range=[0.01, 5.0], resolution=50, length_log_scale=True, gd_points=None, save_figure=False, figure_path=None):
		"""Plot the log marginal likelihood of the process for different hyperparameter values
		if gd_points are given it will plot the evolution as well
		"""
		# save these for reseting at the end
		old_hyperparameters = self.cop.hyperparameters

		# set up the points to plot over
		log_marginal_likelihood = numpy.zeros((resolution, resolution))
		alpha_points = numpy.arange(
				alpha_range[0],
				alpha_range[1],
				(alpha_range[1] - alpha_range[0])/resolution
				)
		if length_log_scale:
			length_points = numpy.arange(
					numpy.log(length_range[0]),
					numpy.log(length_range[1]),
					(numpy.log(length_range[1]) - numpy.log(length_range[0]))/resolution
					)
			length_points = numpy.exp(length_points)
		else:
			length_points = numpy.arange(
					length_range[0],
					length_range[1],
					(length_range[1] - length_range[0])/resolution
					)

		# calculate the marginal likelihood for each pairwise set of hyperparameters
		for alpha_idx, alpha_val in enumerate(alpha_points):
			for length_idx, length_val in enumerate(length_points):
				self.cop.hyperparameters = [alpha_val, length_val]

				log_marginal_likelihood[alpha_idx][length_idx] = self.get_log_marginal_likelihood()

		self.cop.hyperparameters = old_hyperparameters

		# make the grid and the figure
		X, Y = numpy.meshgrid(alpha_points, length_points)
		fig = plt.figure()
		ax = fig.add_subplot(111)

		if gd_points:
			ax.plot(gd_points[0], gd_points[1], 'bx')

		# plot everything and set the axis
		plt.contour(X, Y, log_marginal_likelihood)
		ax.set_title("Marginal Likelihood vs hyperparameters")
		ax.set_xlim(numpy.min(alpha_points), numpy.max(alpha_points))
		ax.set_xlabel('alpha')
		ax.set_ylim(numpy.min(length_points), numpy.max(length_points))
		ax.set_ylabel('length')

		# save the figure
		if save_figure:
			if not figure_path:
				return fig
			else:
				plt.savefig(figure_path, bbox_inches=0) # remove whitespace and save
		else:
			plt.show()


	def plot_contour_and_quiver(self, resolution=20, points_being_sampled=[], save_figure=False, figure_path=None):
		"""Plot the GPP contour plot and grad_mu"""

		if not len(self.domain) == 1:
			raise(ValueError, "Can only plot contour and quiver for 1D functions")

		plotting_points = numpy.arange(self.domain[0][0], self.domain[0][1], (self.domain[0][1] - self.domain[0][0])/float(resolution))

		sample_grad_EI_dx1 = numpy.zeros((resolution, resolution))
		sample_grad_EI_dx2 = numpy.zeros((resolution, resolution))
		sample_EI = numpy.zeros((resolution, resolution))

		for i, x1 in enumerate(plotting_points):
			for j, x2 in enumerate(plotting_points):

				union_of_points = [numpy.array([x1]), numpy.array([x2])]
				union_of_points.extend(points_being_sampled)

				union_of_points_without_x1 = [numpy.array([x2])]
				union_of_points_without_x1.extend(points_being_sampled)

				union_of_points_without_x2 = [numpy.array([x1])]
				union_of_points_without_x2.extend(points_being_sampled)

				sample_grad_EI_dx1[i][j] = self.get_expected_grad_EI(numpy.array([x1]), union_of_points_without_x1)
				sample_grad_EI_dx2[i][j] = self.get_expected_grad_EI(numpy.array([x2]), union_of_points_without_x2)
				sample_EI[i][j] = self.get_expected_improvement(union_of_points)
				print x1,x2

		X, Y = numpy.meshgrid(plotting_points, plotting_points)

		fig = plt.figure()
		ax = fig.add_subplot(111)

		if points_being_sampled:
			ax.plot([numpy.min(plotting_points), numpy.max(plotting_points)], [points_being_sampled[0], points_being_sampled[0]], 'b')
			if len(points_being_sampled) == 2:
				ax.plot([numpy.min(plotting_points), numpy.max(plotting_points)], [points_being_sampled[1], points_being_sampled[1]], 'b')

		plt.quiver(Y, X, sample_grad_EI_dx1, sample_grad_EI_dx2)
		plt.contour(X, Y, sample_EI)
		ax.set_title("Sample EI and grad EI")
		ax.set_xlim(numpy.min(plotting_points), numpy.max(plotting_points))
		ax.set_ylim(numpy.min(plotting_points), numpy.max(plotting_points))

		if save_figure:
			if not figure_path:
				return fig
			else:
				plt.savefig(figure_path, bbox_inches=0) # remove whitespace and save
		else:
			plt.show()
