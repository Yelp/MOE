/*!
  \file gpp_python_gaussian_process.cpp
  \rst
  This file has the logic to construct a GaussianProcess (C++ object) from Python and invoke its member functions.
  The data flow follows the basic 4 step from gpp_python_common.hpp.

  .. Note:: several internal functions of this source file are only called from ``Export*()`` functions,
    so their description, inputs, outputs, etc. comments have been moved. These comments exist in
    ``Export*()`` as Python docstrings, so we saw no need to repeat ourselves.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_gaussian_process.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <vector>  // NOLINT(build/include_order)

#include <boost/python/def.hpp>  // NOLINT(build/include_order)
#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)
#include <boost/python/make_constructor.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_exception.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Surrogate "constructor" for GaussianProcess intended only for use by boost::python.  This aliases the normal C++ constructor,
  replacing ``double const * restrict`` arguments with ``const boost::python::list&`` arguments.
\endrst*/
GaussianProcess * make_gaussian_process(const boost::python::list& hyperparameters,
                                        const boost::python::list& points_sampled,
                                        const boost::python::list& points_sampled_value,
                                        const boost::python::list& noise_variance,
                                        int dim, int num_sampled) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled,
                                                points_sampled_value, noise_variance,
                                                points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha,
                                       input_container.lengths.data());

  GaussianProcess * new_gp = new GaussianProcess(square_exponential, input_container.points_sampled.data(),
                                                 input_container.points_sampled_value.data(),
                                                 input_container.noise_variance.data(),
                                                 input_container.dim, input_container.num_sampled);
  new_gp->SetRandomizedSeed(0);
  return new_gp;
}

boost::python::list GetMeanWrapper(const GaussianProcess& gaussian_process,
                                   const boost::python::list& points_to_sample,
                                   int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_mean(input_container.num_to_sample);
  int num_derivatives = 0;
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeMeanOfPoints(points_to_sample_state, to_sample_mean.data());

  return VectorToPylist(to_sample_mean);
}

boost::python::list GetGradMeanWrapper(const GaussianProcess& gaussian_process,
                                       const boost::python::list& points_to_sample,
                                       int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_grad_mean(input_container.dim*input_container.num_to_sample);
  int num_derivatives = num_to_sample;
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeGradMeanOfPoints(points_to_sample_state, to_sample_grad_mean.data());

  return VectorToPylist(to_sample_grad_mean);
}

boost::python::list GetVarWrapper(const GaussianProcess& gaussian_process,
                                  const boost::python::list& points_to_sample,
                                  int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_var(Square(input_container.num_to_sample));
  int num_derivatives = 0;
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, to_sample_var.data());

  boost::python::list result;

  // copy lower triangle of chol_var into its upper triangle b/c python expects a proper symmetric matrix
  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < i; ++j) {
      to_sample_var[i*num_to_sample + j] = to_sample_var[j*num_to_sample + i];
    }
  }

  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < num_to_sample; ++j) {
      result.append(to_sample_var[j*num_to_sample + i]);
    }
  }

  return result;
}

boost::python::list GetCholVarWrapper(const GaussianProcess& gaussian_process,
                                      const boost::python::list& points_to_sample,
                                      int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> chol_var(Square(input_container.num_to_sample));
  int num_derivatives = 0;
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, chol_var.data());
  int leading_minor = ComputeCholeskyFactorL(num_to_sample, chol_var.data());
  if (unlikely(leading_minor != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample or points_to_sample duplicating points_sampled with 0 noise.", chol_var.data(), num_to_sample, leading_minor);
  }

  boost::python::list result;

  ZeroUpperTriangle(num_to_sample, chol_var.data());
  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < num_to_sample; ++j) {
      result.append(chol_var[j*num_to_sample + i]);
    }
  }

  return result;
}

boost::python::list GetGradVarWrapper(const GaussianProcess& gaussian_process,
                                      const boost::python::list& points_to_sample,
                                      int num_to_sample, int num_derivatives) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_grad_var(input_container.dim*Square(input_container.num_to_sample)*num_derivatives);
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeGradVarianceOfPoints(&points_to_sample_state, to_sample_grad_var.data());

  return VectorToPylist(to_sample_grad_var);
}

boost::python::list GetGradCholVarWrapper(const GaussianProcess& gaussian_process,
                                          const boost::python::list& points_to_sample,
                                          int num_to_sample, int num_derivatives) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_grad_var(input_container.dim*Square(input_container.num_to_sample)*num_derivatives);
  std::vector<double> chol_var(Square(input_container.num_to_sample));
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(),
                                                    input_container.num_to_sample, num_derivatives);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, chol_var.data());
  int leading_minor = ComputeCholeskyFactorL(input_container.num_to_sample, chol_var.data());
  if (unlikely(leading_minor != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample or points_to_sample duplicating points_sampled with 0 noise.", chol_var.data(), num_to_sample, leading_minor);
  }
  gaussian_process.ComputeGradCholeskyVarianceOfPoints(&points_to_sample_state, chol_var.data(),
                                                       to_sample_grad_var.data());

  return VectorToPylist(to_sample_grad_var);
}

void AddPointsToGPWrapper(GaussianProcess * gaussian_process,
                          const boost::python::list& new_points,
                          const boost::python::list& new_points_value,
                          const boost::python::list& new_points_noise_variance,
                          int num_new_points) {
  int dim = gaussian_process->dim();
  std::vector<double> new_points_C(dim*num_new_points);
  std::vector<double> new_points_value_C(num_new_points);
  std::vector<double> new_points_noise_variance_C(num_new_points);

  CopyPylistToVector(new_points, dim*num_new_points, new_points_C);
  CopyPylistToVector(new_points_value, num_new_points, new_points_value_C);
  CopyPylistToVector(new_points_noise_variance, num_new_points, new_points_noise_variance_C);

  gaussian_process->AddPointsToGP(new_points_C.data(), new_points_value_C.data(),
                                  new_points_noise_variance_C.data(), num_new_points);
}

double SamplePointFromGPWrapper(GaussianProcess * gaussian_process,
                                const boost::python::list& point_to_sample,
                                double noise_variance) {
  int num_to_sample = 1;  // we're only drawing 1 point at a time here
  PythonInterfaceInputContainer input_container(point_to_sample, gaussian_process->dim(), num_to_sample);

  return gaussian_process->SamplePointFromGP(input_container.points_to_sample.data(), noise_variance);
}

void PrintHistoricalData(const GaussianProcess& gaussian_process) {
  PrintMatrixTrans(gaussian_process.points_sampled().data(), gaussian_process.num_sampled(), gaussian_process.dim());
  PrintMatrix(gaussian_process.points_sampled_value().data(), 1, gaussian_process.num_sampled());
  PrintMatrix(gaussian_process.noise_variance().data(), 1, gaussian_process.num_sampled());
}

}  // end unnamed namespace

void ExportGaussianProcessFunctions() {
  boost::python::class_<GaussianProcess, boost::noncopyable>("GaussianProcess", boost::python::no_init)
      .def("__init__", boost::python::make_constructor(&make_gaussian_process), R"%%(
    Constructor for a ``GPP.GaussianProcess`` object.

    Seeds internal NormalRNG randomly.

    :param hyperparameters: covariance hyperparameters; see "Details on ..." section at the top of ``BOOST_PYTHON_MODULE``
    :type hyperparameters: list of len 2; index 0 is a float64 ``\alpha`` (signal variance) and index 1 is the length scales (list of floa64 of length ``dim``)
    :param points_sampled: points that have already been sampled
    :type points_sampled: list of float64 with shape (num_sampled, dim)
    :param points_sampled_value: values of the already-sampled points
    :type points_sampled_value: list of float64 with shape (num_sampled, )
    :param noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
    :type noise_variance: list of float64 with shape (num_sampled, )
    :param dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :type param: int > 0
    :param num_sampled: number of already-sampled points
    :type num_sampled: int > 0
          )%%")
      .add_property("dim", &GaussianProcess::dim, "Return the number of spatial dimensions.")
      .add_property("num_sampled", &GaussianProcess::num_sampled, "Return the number of sampled points.")
      .def("compute_mean_of_points", GetMeanWrapper, R"%%(
        Compute the (predicted) mean, mus, of the Gaussian Process posterior.
        ``mus_i = Ks_{i,k} * K^-1_{k,l} * y_l = Ks^T * K^-1 * y``

        :param points_to_sample: points at which to compute GP-derived quantities (mean, variance, etc.; i.e., make predictions)
        :type points_to_sample: list of float64 with shape (num_to_sample, dim)
        :param num_to_sample: number of points to sample
        :type num_to_sample: int > 0
        :return: GP mean evaluated at each of ``points_to_sample``
        :rtype: list of float64 with shape (num_to_sample, )
        )%%")
      .def("compute_grad_mean_of_points", GetGradMeanWrapper, R"%%(
        Compute the gradient of the (predicted) mean, ``mus``, of the Gaussian Process posterior.
        Gradient is computed wrt each point in points_to_sample.
        Known zero terms are dropped (see below).
        ``mus_i = Ks_{i,k} * K^-1_{k,l} * y_l = Ks^T * K^-1 * y``
        In principle, we compute ``\pderiv{mus_i}{Xs_{d,p}}``
        But this is zero unless ``p == i``. So this method only returns the "block diagonal."

        :param points_to_sample: points at which to compute GP-derived quantities (mean, variance, etc.; i.e., make predictions)
        :type points_to_sample: list of float64 with shape (num_to_sample, dim)
        :param num_to_sample: number of points to sample
        :type num_to_sample: int > 0
        :return: gradient of the mean of the GP.  ``grad_mu[d][i]`` is
            actually the gradient of ``\mu_i`` with respect to ``x_{d,i}``, the d-th dimension of
            the i-th entry of ``points_to_sample``.
        :rtype: list of float64 with shape (num_to_sample, dim)
        )%%")
      .def("compute_variance_of_points", GetVarWrapper, R"%%(
        Compute the (predicted) variance, ``Vars``, of the Gaussian Process posterior.
        ``L * L^T = K``
        ``V = L^-1 * Ks^T``
        ``Vars = Kss - (V^T * V)``
        Expanded index notation:
        ``Vars_{i,j} = Kss_{i,j} - Ks_{i,l} * K^-1_{l,k} * Ks_{k,j} = Kss - Ks^T * K^-1 * Ks`

        .. Note:: ``Vars`` is symmetric (in fact, SPD).

        :param points_to_sample: points at which to compute GP-derived quantities (mean, variance, etc.; i.e., make predictions)
        :type points_to_sample: list of float64 with shape (num_to_sample, dim)
        :param num_to_sample: number of points to sample
        :type num_to_sample: int > 0
        :return: GP variance evaluated at ``points_to_sample``,
            ordered as num_to_sample rows of length num_to_sample
        :rtype: list of float64 with shape (num_to_sample, num_to_sample)
        )%%")
      .def("compute_cholesky_variance_of_points", GetCholVarWrapper, R"%%(
        Computes the Cholesky Decomposition of the predicted GP variance:
        ``L * L^T = Vars``, where Vars is the output of get_var().
        See that function's docstring for further details.

        :param points_to_sample: points at which to compute GP-derived quantities (mean, variance, etc.; i.e., make predictions)
        :type points_to_sample: list of float64 with shape (num_to_sample, dim)
        :param num_to_sample: number of points to sample
        :type num_to_sample: int > 0
        :return: cholesky factor (L) of the GP variance evaluated at ``points_to_sample``,
            ordered as num_to_sample rows of length num_to_sample
        :rtype: list of float64 with shape (num_to_sample, num_to_sample)
        )%%")
      .def("compute_grad_variance_of_points", GetGradVarWrapper, R"%%(
        Similar to get_grad_chol_var() except this does not include the gradient terms from
        the cholesky factorization.  Description will not be duplicated here.
        )%%")
      .def("compute_grad_cholesky_variance_of_points", GetGradCholVarWrapper, R"%%(
        Compute gradient of the Cholesky Factorization of the (predicted) variance, Vars, of the Gaussian Process posterior.
        Gradient is computed wrt points_to_sample[0:num_derivatives].
        ``L * L^T = K``
        ``V = L^-1 * Ks^T``
        ``Vars = Kss - (V^T * V)``
        Expanded index notation:
        ``Vars_{i,j} = Kss_{i,j} - Ks_{i,l} * K^-1_{l,k} * Ks_{k,j} = Kss - Ks^T * K^-1 * Ks``
        Then the Cholesky Decomposition is ``Ls * Ls^T = Vars``

        General derivative expression: ``\pderiv{Ls_{i,j}}{Xs_{d,p}}``
        We compute this for ``p = 0, 1, ..., num_derivatives-1``.

        :param points_to_sample: points at which to compute GP-derived quantities (mean, variance, etc.; i.e., make predictions)
        :type points_to_sample: list of float64 with shape (num_to_sample, dim)
        :param num_to_sample: number of points to sample
        :type num_to_sample: int > 0
        :param num_derivatives: return derivatives wrt ``points_to_sample[0:num_derivatives]`
        :type num_derivatives: 0 < int <= num_to_sample
        :return: gradient of the cholesky-factored variance of the GP.
          ``grad_chol[k][j][i][d]`` is actually the gradients of ``var_{i,j}`` with
          respect to ``x_{d,k}``, the d-th dimension of the k-th entry of ``points_to_sample``
        :rtype: list of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)
      )%%")
      .def("add_sampled_points", AddPointsToGPWrapper, R"%%(
        Add the specified (point, fcn value, noise variance) historical data to this GP.

        Forces recomputation of all derived quantities for GP to remain consistent.

        :param new_points: coordinates of each new point to add
        :type new_points: list of float64 with shape (num_new_points, dim)
        :param new_points_value: function value at each new point
        :type new_points_value: list of float64 with shape (num_new_points, )
        :param new_points_noise_variance: \sigma_n^2 corresponding to the signal noise in measuring new_points_value
        :type new_points_noise_variance: list of float64 with shape (num_new_points, )
        :param num_new_points: number of new points to add to the GP
        :type num_new_points: int
      )%%")
      .def("sample_point_from_gp", SamplePointFromGPWrapper, R"%%(
        Sample a function value from a Gaussian Process prior, provided a point at which to sample.

        Uses the formula ``function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2``, where ``w1, w2``
        are draws from \ms N(0,1)\me.

        :param point_to_sample: coordinates of the point at which to generate a function value (from GP)
        :type point_to_sample: list of float64 with shape (dim, )
        :param noise_variance_this_point: if this point is to be added into the GP, it needs to be generated with
          its associated noise var
        :type noise_variance_this_point: float64 >= 0.0
        :return: function value drawn from this GP
        :rtype: float64
      )%%")
      .def("set_explicit_seed", &GaussianProcess::SetExplicitSeed, "Seed the internal RNG with the specified seed.")
      .def("set_randomized_seed", &GaussianProcess::SetRandomizedSeed, R"%%(
        Seed the internal RNG with a combination of the specified seed and other factors.

        See gpp_random, struct NormalRNG for details.
      )%%")
      .def("reset_to_most_recent_seed", &GaussianProcess::ResetToMostRecentSeed, "Seed the internal RNG with the last used seed.")
      .def("print_historical_data", PrintHistoricalData)
      ;  // NOLINT, this is boost style
}

}  // end namespace optimal_learning
