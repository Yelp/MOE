# -*- coding: utf-8 -*-
r"""Interfaces for structures needed by the optimal_learning package to build Gaussian Process models and optimize the Expected Improvement.

The package comments here will introduce what optimal_learning is attempting to accomplish, provide an outline of Gaussian Processes,
and introduce the notion of Expected Improvement and its optimization.

The modules in this package provide the interface with interacting with all the features of optimal_learning.

.. NOTE:: These comments were copied from the file comments in gpp_math.hpp.

  At a high level, this file optimizes an objective function \ms f(x)\me.  This operation
  requires data/uncertainties about prior and concurrent experiments as well as
  a covariance function describing how these data [are expected to] relate to each
  other.  The points x represent experiments. If \ms f(x)\me is say, survival rate for
  a drug test, the dimensions of x might include dosage amount, dosage frequency,
  and overall drug-use time-span.

  The objective function is not required in closed form; instead, only the ability
  to sample it at points of interest is needed.  Thus, the optimization process
  cannot work with \ms f(x)\me directly; instead a surrogate is built via interpolation
  with Gaussian Proccesses (GPs).

  Following Rasmussen & Williams (2.2), a Gaussian Process is a collection of random
  variables, any finite number of which have a joint Gaussian distribution (Defn 2.1).
  Hence a GP is fully specified by its mean function, \ms m(x)\me, and covariance function,
  \ms k(x,x')\me.  Then we assume that a real process \ms f(x)\me (e.g., drug survival rate) is
  distributed like:

  .. math:: f(x) ~ GP(m(x), k(x,x'))

  with

  .. math:: m(x) = E[f(x)], k(x,x') = E[(f(x) - m(x))*(f(x') - m(x'))].

  Then sampling from \ms f(x)\me is simply drawing from a Gaussian with the appropriate mean
  and variance.

  However, since we do not know \ms f(x)\me, we cannot precisely build its corresponding GP.
  Instead, using samples from \ms f(x)\me (e.g., by measuring experimental outcomes), we can
  iteratively improve our estimate of \ms f(x)\me.  See GaussianProcessInterface class docs
  and implementation docs for details on how this is done.

  The optimization process models the objective using a Gaussian process (GP) prior
  (also called a GP predictor) based on the specified covariance and the input
  data (e.g., through member functions compute_mean_of_points, compute_variance_of_points).  Using the GP,
  we can compute the expected improvement (EI) from sampling any particular point.  EI
  is defined relative to the best currently known value, and it represents what the
  algorithm believes is the most likely outcome from sampling a particular point in parameter
  space (aka conducting a particular experiment).

  See ExpectedImprovementInterface ABC and implementation docs for further details on computing EI.
  Both support compute_expected_improvement() and compute_grad_expected_improvement().

  The dimension of the GP is equal to the number of simultaneous experiments being run;
  i.e., the GP may be multivariate.  The behavior of the GP is controlled by its underlying
  covariance function and the data/uncertainty of prior points (experiments).

  With the ability the compute EI, the final step is to optimize
  to find the best EI.  This is done using multistart gradient descent (MGD), in
  multistart_expected_improvement_optimization(). This method wraps a MGD call and falls back on random search
  if that fails. See gpp_optimization.hpp for multistart/optimization templates. This method
  can evaluate and optimize EI at serval points simultaneously; e.g., if we wanted to run 4 simultaneous
  experiments, we can use EI to select all 4 points at once.

  The literature (e.g., Ginsbourger 2008) refers to these problems collectively as q-EI, where q
  is a positive integer. So 1-EI is the originally dicussed usage, and the previous scenario with
  multiple simultaneous points/experiments would be called 4-EI.

  Additionally, there are use cases where we have existing experiments that are not yet complete but
  we have an opportunity to start some new trials. For example, maybe we are a drug company currently
  testing 2 combinations of dosage levels. We got some new funding, and can now afford to test
  3 more sets of dosage parameters. Ideally, the decision on the new experiments should depend on
  the existence of the 2 ongoing tests. We may not have any data from the ongoing experiments yet;
  e.g., they are [double]-blind trials. If nothing else, we would not want to duplicate any
  existing experiments! So we want to solve 3-EI using the knowledge of the 2 ongoing experiments.

  We call this q,p-EI, so the previous example would be 3,2-EI. The q-EI notation is equivalent to
  q,0-EI; if we do not explicitly write the value of p, it is 0. So q is the number of new
  (simultaneous) experiments to select. In code, this would be the size of the output from EI
  optimization (i.e., ``best_points_to_sample``, of which there are ``q = num_to_sample points``).
  p is the number of ongoing/incomplete experiments to take into account (i.e., ``points_being_sampled``
  of which there are ``p = num_being_sampled`` points).

  Back to optimization: the idea behind gradient descent is simple.  The gradient gives us the
  direction of steepest ascent (negative gradient is steepest descent).  So each iteration, we
  compute the gradient and take a step in that direction.  The size of the step is not specified
  by GD and is left to the specific implementation.  Basically if we take steps that are
  too large, we run the risk of over-shooting the solution and even diverging.  If we
  take steps that are too small, it may take an intractably long time to reach the solution.
  Thus the magic is in choosing the step size; we do not claim that our implementation is
  perfect, but it seems to work reasonably.  See ``gpp_optimization.hpp`` for more details about
  GD as well as the template definition.

  For particularly difficult problems or problems where gradient descent's parameters are not
  well-chosen, GD can fail to converge.  If this happens, we can fall back on heuristics;
  e.g., 'dumb' search (i.e., evaluate EI at a large number of random points and take the best
  one). This functionality is accessed through: multistart_expected_improvement_optimization().

  And domain-specific notation, following Rasmussen, Williams:
    * ``X = points_sampled``; this is the training data (size ``dim`` X ``num_sampled``), also called the design matrix
    * ``Xs = points_to_sample``; this is the test data (size ``dim`` X num_to_sample``)
    * ``y, f, f(x) = points_sampled_value``, the experimental results from sampling training points
    * ``K, K_{ij}, K(X,X) = covariance(X_i, X_j)``, covariance matrix between training inputs (``num_sampled x num_sampled``)
    * ``Ks, Ks_{ij}, K(X,Xs) = covariance(X_i, Xs_j)``, covariance matrix between training and test inputs (``num_sampled x num_to_sample``)
    * ``Kss, Kss_{ij}, K(Xs,Xs) = covariance(Xs_i, Xs_j)``, covariance matrix between test inputs (``num_to_sample x num_to_sample``)
    * ``\theta``: (vector) of hyperparameters for a covariance function

  .. NOTE::
       Due to confusion with multiplication (K_* looks awkward in code comments), Rasmussen & Williams' \ms K_*\me
       notation has been repalced with ``Ks`` and \ms K_{**}\me is ``Kss``.

  Connecting to the q,p-EI notation, both the points represented by "q" and "p" are represented by ``Xs``. Within
  the GP, there is no distinction between points being sampled by ongoing experiments and new points to sample.

"""
