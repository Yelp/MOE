# -*- coding: utf-8 -*-
r"""Interface for covariance function: covariance of two points and spatial/hyperparameter derivatives.

.. Note:: comments are copied from the file comments of gpp_covariance.hpp

Covariance functions have a few fundamental properties (see references at the bottom for full details).  In short,
they are SPSD (symmetric positive semi-definite): ``k(x,x') = k(x', x)`` for any ``x,x'`` and ``k(x,x) >= 0`` for all ``x``.
As a consequence, covariance matrices are SPD as long as the input points are all distinct.

Additionally, the Square Exponential and Matern covariances (as well as other functions) are stationary. In essence,
this means they can be written as ``k(r) = k(|x - x'|) = k(x, x') = k(x', x)``.  So they operate on distances between
points as opposed to the points themselves.  The name stationary arises because the covariance is the same
modulo linear shifts: ``k(x+a, x'+a) = k(x, x').``

Covariance functions are a fundamental component of gaussian processes: as noted in the gpp_math.hpp header comments,
gaussian processes are defined by a mean function and a covariance function.  Covariance functions describe how
two random variables change in relation to each other--more explicitly, in a GP they specify how similar two points are.
The choice of covariance function is important because it encodes our assumptions about how the "world" behaves.

Covariance functions also generally have hyperparameters (e.g., signal/background noise, length scales) that specify the
assumed behavior of the Gaussian Process. Specifying hyperparameters is tricky because changing them fundamentally changes
the behavior of the GP. :mod:`moe.optimal_learning.python.interfaces.optimization_interface` together
with :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface` provide methods
optimizing and evaluating model fit, respectively.

"""
from builtins import object
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass


class CovarianceInterface(with_metaclass(ABCMeta, object)):

    r"""Interface for a covariance function: covariance of two points and spatial/hyperparameter derivatives.

    .. Note:: comments are copied from the class comments of CovarianceInterface in gpp_covariance.hpp

    Abstract class to enable evaluation of covariance functions--supports the evaluation of the covariance between two
    points, as well as the gradient with respect to those coordinates and gradient/hessian with respect to the
    hyperparameters of the covariance function.

    Covariance operaters, ``cov(x_1, x_2)`` are SPD.  Due to the symmetry, there is no need to differentiate wrt x_1 and x_2; hence
    the gradient operation should only take gradients wrt dim variables, where ``dim = |x_1|``

    Hyperparameters (denoted ``\theta_j``) are stored as class member data by subclasses.

    Implementers of this ABC are required to manage their own hyperparameters.

    TODO(GH-71): getter/setter for hyperparameters.

    """

    @abstractproperty
    def num_hyperparameters(self):
        """Return the number of hyperparameters of this covariance function."""
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
    def covariance(self, point_one, point_two):
        r"""Compute the covariance function of two points, cov(``point_one``, ``point_two``).

        .. Note:: comments are copied from the matching method comments of CovarianceInterface in gpp_covariance.hpp
          and comments are copied to the matching method comments of
          :mod:`moe.optimal_learning.python.python_version.covariance.SquareExponential`.

        The covariance function is guaranteed to be symmetric by definition: ``covariance(x, y) = covariance(y, x)``.
        This function is also positive definite by definition.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: value of covariance between the input points
        :rtype: float64

        """
        pass

    @abstractmethod
    def grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to the FIRST argument, point_one.

        .. Note:: comments are copied from the matching method comments of CovarianceInterface in gpp_covariance.hpp
          and comments are copied to the matching method comments of
          :mod:`moe.optimal_learning.python.python_version.covariance.SquareExponential`.

        This distinction is important for maintaining the desired symmetry.  ``Cov(x, y) = Cov(y, x)``.
        Additionally, ``\pderiv{Cov(x, y)}{x} = \pderiv{Cov(y, x)}{x}``.
        However, in general, ``\pderiv{Cov(x, y)}{x} != \pderiv{Cov(y, x)}{y}`` (NOT equal!  These may differ by a negative sign)

        Hence to avoid separate implementations for differentiating against first vs second argument, this function only handles
        differentiation against the first argument.  If you need ``\pderiv{Cov(y, x)}{x}``, just swap points x and y.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: grad_cov: i-th entry is ``\pderiv{cov(x_1, x_2)}{x_i}``
        :rtype: array of float64 with shape (dim)

        """
        pass

    @abstractmethod
    def hyperparameter_grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to its hyperparameters.

        .. Note:: comments are copied from the matching method comments of CovarianceInterface in gpp_covariance.hpp
          and comments are copied to the matching method comments of
          :mod:`moe.optimal_learning.python.python_version.covariance.SquareExponential`.

        Unlike GradCovariance(), the order of point_one and point_two is irrelevant here (since we are not differentiating against
        either of them).  Thus the matrix of grad covariances (wrt hyperparameters) is symmetric.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: grad_hyperparameter_cov: i-th entry is ``\pderiv{cov(x_1, x_2)}{\theta_i}``
        :rtype: array of float64 with shape (num_hyperparameters)

        """
        pass

    @abstractmethod
    def hyperparameter_hessian_covariance(self, point_one, point_two):
        r"""Compute the hessian of self.covariance(point_one, point_two) with respect to its hyperparameters.

        .. Note:: comments are copied from the matching method comments of CovarianceInterface in gpp_covariance.hpp

        The Hessian matrix of the covariance evaluated at x_1, x_2 with respect to the hyperparameters.  The Hessian is defined as:
        ``[ \ppderiv{cov}{\theta_0^2}              \mixpderiv{cov}{\theta_0}{\theta_1}    ... \mixpderiv{cov}{\theta_0}{\theta_{n-1}} ]``
        ``[ \mixpderiv{cov}{\theta_1}{\theta_0}    \ppderiv{cov}{\theta_1^2 }             ... \mixpderiv{cov}{\theta_1}{\theta_{n-1}} ]``
        ``[      ...                                                                                     ...                          ]``
        ``[ \mixpderiv{cov}{\theta_{n-1}{\theta_0} \mixpderiv{cov}{\theta_{n-1}{\theta_1} ... \ppderiv{cov}{\theta_{n-1}^2}           ]``
        where "cov" abbreviates covariance(x_1, x_2) and "n" refers to the number of hyperparameters.

        Unless noted otherwise in subclasses, the Hessian is symmetric (due to the equality of mixed derivatives when a function
        f is twice continuously differentiable).

        Similarly to the gradients, the Hessian is independent of the order of ``x_1, x_2: H_{cov}(x_1, x_2) = H_{cov}(x_2, x_1)``

        For further details: http://en.wikipedia.org/wiki/Hessian_matrix

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape(dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: hessian_hyperparameter_cov: ``(i,j)``-th entry is ``\mixpderiv{cov(x_1, x_2)}{\theta_i}{\theta_j}``
        :rtype: array of float64 with shape (num_hyperparameters, num_hyperparameters)

        """
        pass
