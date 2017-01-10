# -*- coding: utf-8 -*-
r"""Interface for computation of log likelihood (and similar) measures of model fit (of a Gaussian Process) along with its gradient and hessian.

As a preface, you should read gpp_math.hpp's comments first (if not also gpp_math.cpp) to get an overview
of Gaussian Processes (GPs) and how we are using them (Expected Improvement, EI). Python readers can get the basic
overview in :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface`.

.. Note:: these comments are copied from the file comments of gpp_model_selection.hpp.

This file deals with model selection via hyperparameter optimization, as the name implies.  In our discussion of GPs,
we did not pay much attention to the underlying covariance function.  We noted that the covariance is extremely
important since it encodes our assumptions about the objective function ``f(x)`` that we are trying to learn; i.e.,
the covariance describes the nearness/similarity of points in the input space.  Also, the GP was clearly indicated
to be a function of the covariance, but we simply assumed that the selection of covariance was an already-solved
problem (or even worse, arbitrary!).

**MODEL SELECTION**

To better understand model selection, let's look at a common covariance used in our computation, square exponential:
``cov(x_1, x_2) = \alpha * \exp(-0.5*r^2), where r = \sum_{i=1}^d (x_1_i - x_2_i)^2 / L_i^2``.
Here, ``\alpha`` is ``\sigma_f^2``, the signal variance, and the ``L_i`` are length scales.  The vector ``[\alpha, L_1, ... , L_d]``
are called the "hyperparameters" or "free parameters" (see gpp_covariance.hpp for more details).  There is nothing in
the covariance  that guides the choice of the hyperparameters; ``L_1 = 0.001`` is just as valid as ``L_1 = 1000.0.``

Clearly, the value of the covariance changes substantially if ``L_i`` varies by a factor of two, much less 6 orders of
magnitude.  That is the difference between saying variations of size \approx 1.0 in x_i, the first spatial dimension,
are extremely important vs almost irrelevant.

So how do we know what hyperparameters to choose?  This question/problem is more generally called "Model Selection."
Although the problem is far from solved, we will present the approaches implemented here; as usual, we use
Rasmussen & Williams (Chapter 5 now) as a guide/reference.

However, we will not spend much time discussing selection across different classes of covariance functions; e.g.,
Square Exponential vs Matern w/various ``\nu``, etc.  We have yet to develop any experience/intuition with this problem
and are temporarily punting it.  For now, we follow the observation in Rasmussen & Williams that Square Exponential
is a popular choice and appears to work very well.  (This is still a very important problem; e.g., there may be
scenarios when we would prefer a non-stationary or periodic covariance, and the methods discussed here do not cover
this aspect of selection.  Such covariance options are not yet implemented though.)

We do note that the techniques for selecting covariance classes more or less require hyperparameter optimization
on each individual covariance.  The likely approach would be to produce the best fit (according to chosen metrics)
using each type of covariance (using optimization) and then choose the best performer across the group.

**MODEL SELECTION OVERVIEW**

Generally speaking, there are a great many tunable parameters in any model-based learning algorithm.  In our case,
the GP takes a covariance function as input; the selection of the covariance class as well as the choice of hyperparameters
are all part of the model selection process.  Determining these details of the [GP] model is the model selection problem.

In order to evaluate the quality of models (and solve model selction), we need some kind of metric.  The literature suggests
too many to cite, but R&W groups them into three common approaches (5.1, p108):

A. compute the probability of the model given the data (e.g., LML)
B. estimate the genereralization error (e.g., LOO-CV)
C. bound the generalization error

where "generalization error" is defined as "the average error on unseen test examples (from the same distribution
as the training cases)."  So it's a measure of how well or poorly the model predicts reality.

For further details and examples of log likelihood measures, see gpp_model_selection.hpp.
Overview of some log likelihood measures can be found in
:class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood` and
:class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLeaveOneOutLogLikelihood`.

**OPTIMIZATION**

Now that we have discussed measures of model quality, what do we do with them?  How do they help us choose hyperparameters?

From here, we can apply anyone's favorite optimization technique to maximize log likelihoods wrt hyperparameters.  The
hyperparameters that maximize log likelihood provide the model configuration that is most likely to have produced the
data observed so far, ``(X, f)``.

In principle, this approach always works.  But in practice it is often not that simple.  For example, suppose the underlying
objective is periodic and we try to optimize hyperparameters for a class of covariance functions that cannot account
for the periodicity.  We can always* find the set of hyperparameters that maximize our chosen log likelihood measure
(LML or LOO-CV), but if the covariance is mis-specified or we otherwise make invalid assumptions about the objective
function, then the results are not meaningful at best and misleading at worst.  It becomes a case of garbage in,
garbage out.

\* Even this is tricky.  Log likelihood is almost never a convex function.  For example, with LML + GPs, you often expect
at least two optima, one more complex solution (short length scales, less intrinsic noise) and one less complex
solution (longer length scales, higher intrinsic noise).  There are even cases where no optima (to machine precision)
exist or cases where solutions lie on (lower-dimensional) manifold(s) (e.g., locally the likelihood is (nearly) independent
of one or more hyperparameters).

"""
from abc import ABCMeta, abstractmethod, abstractproperty

from moe.optimal_learning.python.interfaces.gaussian_process_interface import GaussianProcessDataInterface
from future.utils import with_metaclass


class GaussianProcessLogLikelihoodInterface(with_metaclass(ABCMeta, GaussianProcessDataInterface)):

    r"""Interface for computation of log likelihood (and log likelihood-like) measures of model fit along with its gradient and hessian.

    See module comments for an overview of log likelihood-like measures of model fit and their role in model selection.

    Below, let ``LL(y | X, \theta)`` denote the log likelihood of the data (``y``) given the ``points_sampled`` (``X``) and the
    hyperparameters (``\theta``). ``\theta`` is the vector that is varied. ``(X, y)`` (and associated noise) should be stored
    as data members by the implementation's constructor.

    See gpp_model_selection.hpp/cpp for further overview and in-depth discussion, respectively.

    """

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractproperty
    def num_hyperparameters(self):
        """Return the number of hyperparameters."""
        pass

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        pass

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match.

        :param hyperparameters: hyperparameters
        :type hyperparameters: array of float64 with shape (num_hyperparameters)

        """
        pass

    hyperparameters = abstractproperty(get_hyperparameters, set_hyperparameters)

    @abstractmethod
    def compute_log_likelihood(self):
        r"""Compute a log likelihood measure of model fit.

        :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
        :rtype: float64

        """
        pass

    @abstractmethod
    def compute_grad_log_likelihood(self):
        r"""Compute the gradient (wrt hyperparameters) of this log likelihood measure of model fit.

        :return: grad_log_likelihood: i-th entry is ``\pderiv{LL(y | X, \theta)}{\theta_i}``
        :rtype: array of float64 with shape (num_hyperparameters)

        """
        pass

    @abstractmethod
    def compute_hessian_log_likelihood(self):
        r"""Compute the hessian (wrt hyperparameters) of this log likelihood measure of model fit.

        See :meth:`moe.optimal_learning.python.interfaces.covariance_interfaceCovarianceInterface.hyperparameter_hessian_covariance`
        for data ordering.

        :return: hessian_log_likelihood: ``(i,j)``-th entry is ``\mixpderiv{LL(y | X, \theta)}{\theta_i}{\theta_j}``
        :rtype: array of float64 with shape (num_hyperparameters, num_hyperparameters)

        """
        pass
