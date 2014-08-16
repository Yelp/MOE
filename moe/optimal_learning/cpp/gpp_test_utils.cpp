/*!
  \file gpp_test_utils.cpp
  \rst
  Implementations of utitilies useful for unit testing.
\endrst*/

#include "gpp_test_utils.hpp"

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"

// Debugging macro for use with PingDerivative() in this file.
// Define this macro to make PingDerivative() very verbose, printing details
// about all points checked. This is helpful when implementing, debugging
// and checking gradient code with PingDerivative(). See that function
// for more details.
#ifdef OL_PING_TEST_DEBUG_PRINT
#define OL_PING_TEST_DEBUG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define OL_PING_TEST_DEBUG_PRINTF(...) (void)0
#endif

namespace optimal_learning {

MockExpectedImprovementEnvironment::MockExpectedImprovementEnvironment()
    : dim(-1),
      num_sampled(-1),
      num_to_sample(-1),
      num_being_sampled(-1),
      points_sampled_(20*4),
      points_sampled_value_(20),
      points_to_sample_(4),
      points_being_sampled_(20*4),
      uniform_generator_(kDefaultSeed),
      uniform_double_(range_min, range_max) {
}

void MockExpectedImprovementEnvironment::Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in, int num_sampled_in, UniformRandomGenerator * uniform_generator) {
  if (dim_in != dim || num_to_sample_in != num_to_sample || num_being_sampled_in != num_being_sampled || num_sampled_in != num_sampled) {
    dim = dim_in;
    num_to_sample = num_to_sample_in;
    num_being_sampled = num_being_sampled_in;
    num_sampled = num_sampled_in;

    points_sampled_.resize(num_sampled*dim);
    points_sampled_value_.resize(num_sampled);
    points_to_sample_.resize(num_to_sample*dim);
    points_being_sampled_.resize(num_being_sampled*dim);
  }

  for (int i = 0; i < dim*num_sampled; ++i) {
    points_sampled_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < num_sampled; ++i) {
    points_sampled_value_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < dim*num_to_sample; ++i) {
    points_to_sample_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < dim*num_being_sampled; ++i) {
    points_being_sampled_[i] = uniform_double_(uniform_generator->engine);
  }
}

template <typename DomainType>
MockGaussianProcessPriorData<DomainType>::MockGaussianProcessPriorData(const CovarianceInterface& covariance, const std::vector<double>& noise_variance_in, int dim_in, int num_sampled_in)
    : dim(dim_in),
      num_sampled(num_sampled_in),
      domain_bounds(dim),
      covariance_ptr(covariance.Clone()),
      hyperparameters(covariance_ptr->GetNumberOfHyperparameters()),
      noise_variance(noise_variance_in),
      best_so_far(0.0) {
}

template <typename DomainType>
MockGaussianProcessPriorData<DomainType>::MockGaussianProcessPriorData(const CovarianceInterface& covariance, const std::vector<double>& noise_variance_in, int dim_in, int num_sampled_in, const boost::uniform_real<double>& uniform_double_domain_lower, const boost::uniform_real<double>& uniform_double_domain_upper, const boost::uniform_real<double>& uniform_double_hyperparameters, UniformRandomGenerator * uniform_generator)
    : MockGaussianProcessPriorData<DomainType>(covariance, noise_variance_in, dim_in, num_sampled_in) {
  InitializeHyperparameters(uniform_double_hyperparameters, uniform_generator);
  InitializeDomain(uniform_double_domain_lower, uniform_double_domain_upper, uniform_generator);
  InitializeGaussianProcess(uniform_generator);
}

template <typename DomainType>
MockGaussianProcessPriorData<DomainType>::~MockGaussianProcessPriorData() = default;

template <typename DomainType>
void MockGaussianProcessPriorData<DomainType>::InitializeHyperparameters(const boost::uniform_real<double>& uniform_double_hyperparameters, UniformRandomGenerator * uniform_generator) {
  FillRandomCovarianceHyperparameters(uniform_double_hyperparameters, uniform_generator, &hyperparameters, covariance_ptr.get());
}

template <typename DomainType>
void MockGaussianProcessPriorData<DomainType>::InitializeDomain(const boost::uniform_real<double>& uniform_double_domain_lower, const boost::uniform_real<double>& uniform_double_domain_upper, UniformRandomGenerator * uniform_generator) {
  FillRandomDomainBounds(uniform_double_domain_lower, uniform_double_domain_upper, uniform_generator, &domain_bounds);
  domain_ptr.reset(new DomainType(domain_bounds.data(), dim));
}

template <typename DomainType>
void MockGaussianProcessPriorData<DomainType>::InitializeGaussianProcess(UniformRandomGenerator * uniform_generator) {
  covariance_ptr->SetHyperparameters(hyperparameters.data());

  std::vector<double> points_sampled(dim*num_sampled);
  std::vector<double> points_sampled_value(num_sampled);
  gaussian_process_ptr.reset(new GaussianProcess(*covariance_ptr, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0));

  // generate the "world"
  num_sampled = domain_ptr->GenerateUniformPointsInDomain(num_sampled, uniform_generator, points_sampled.data());
  FillRandomGaussianProcess(points_sampled.data(), noise_variance.data(), dim, num_sampled, points_sampled_value.data(), gaussian_process_ptr.get());
  best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());
}

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
template struct MockGaussianProcessPriorData<TensorProductDomain>;
template struct MockGaussianProcessPriorData<SimplexIntersectTensorProductDomain>;

bool CheckIntEquals(int64_t value, int64_t truth) noexcept {
  bool passed = value == truth;

  if (passed == false) {
    OL_ERROR_PRINTF("value = %lld, truth = %lld, diff = %lld\n", value, truth, truth-value);
  }
  return passed;
}

/*!\rst
  ``\|b - A*x\|_2``

  The quantity ``b - A*x`` is called the "residual."  This is meaningful when ``x`` is
  the solution of the linear system ``A*x = b``.  Then having a small residual norm
  is a *NECESSARY* but *NOT SUFFICIENT* indicator of accuracy in ``x``; that is,
  these quantities need not be small simultaneously.  In particular, we know:

  ``\|\delta x\| / \|x\| \le cond(A) * \|r\| / (\|A\| * \|x\|)``

  where ``x`` is the approximate solution and \delta x is the error.  So
  ``\delta x = 0 <=> r = 0`` BUT if not identically 0, ||r|| can be much larger.

  However, a numerically (backward) stable algorithm will compute solutions
  with small relative residual norms *regardless* of conditioning.  Hence
  coupled with knowledge of the particular algorithm for solving ``A*x = b``,
  residual norm is a valuable measure of correctness.

  Suppose ``x`` (computed solution) satisfies: ``(A + \delta A)*x = b``.  Then:

  ``\|r\| / (\|A\| * \|x\|) \le \|\delta A\| / \|A\|``

  So large ``\|r\|`` indicates a large backward error, implying the linear solver
  is not backward stable (and hence should not be used).
\endrst*/
double ResidualNorm(double const * restrict A, double const * restrict x, double const * restrict b, int size) noexcept {
  std::vector<double> y(b, b + size);  // y = b
  GeneralMatrixVectorMultiply(A, 'N', x, -1.0, 1.0, size, size, size, y.data());  // y -= A * x

  double norm = VectorNorm(y.data(), size);
  return norm;
}

bool CheckDoubleWithin(double value, double truth, double tolerance) noexcept {
  double diff = std::fabs(value - truth);
  bool passed = diff <= tolerance;

  if (passed != true) {
    OL_ERROR_PRINTF("value = %.18E, truth = %.18E, diff = %.18E, tol = %.18E\n", value, truth, diff, tolerance);
  }
  return passed;
}

bool CheckDoubleWithinRelativeWithThreshold(double value, double truth, double tolerance, double threshold) noexcept {
  double denom = std::fabs(truth);
  if (denom < threshold) {
    denom = 1.0;  // don't divide by 0
  }
  double diff = std::fabs((value - truth)/denom);
  bool passed = diff <= tolerance;
  if (passed != true) {
    OL_ERROR_PRINTF("value = %.18E, truth = %.18E, diff = %.18E, tol = %.18E\n", value, truth, diff, tolerance);
  }
  return passed;
}

bool CheckDoubleWithinRelative(double value, double truth, double tolerance) noexcept {
  return CheckDoubleWithinRelativeWithThreshold(value, truth, tolerance, std::numeric_limits<double>::min());
}

/*!\rst
  Uses the Frobenius Norm for convenience; matrix 2-norms are expensive to compute.
\endrst*/
bool CheckMatrixNormWithin(double const * restrict matrix1, double const * restrict matrix2, int size_m, int size_n, double tolerance) noexcept {
  std::vector<double> difference_matrix(matrix1, matrix1+size_m*size_n);

  VectorAXPY(size_m*size_n, -1.0, matrix2, difference_matrix.data());
  double norm = VectorNorm(difference_matrix.data(), size_m*size_n);  // Frobenius norm
  return norm <= tolerance;  // should this be adjusted to remove factor of size_m*size_n?
}

int CheckPointsAreDistinct(double const * restrict point_list, int num_points, int dim, double tolerance) noexcept {
  int num_errors = 0;
  std::vector<double> temp_point(dim);
  for (int i = 0; i < num_points; ++i) {
    for (int j = i + 1; j < num_points; ++j) {
      // copy over i-th point
      std::copy(point_list + i*dim, point_list + (i+1)*dim, temp_point.data());
      // subtract out j-th point
      VectorAXPY(dim, -1.0, point_list + j*dim, temp_point.data());
      // norm to compute distance
      double distance = VectorNorm(temp_point.data(), dim);
      if (distance <= tolerance) {
        ++num_errors;
      }
    }
  }

  return num_errors;
}

namespace {

/*!\rst
  Computes the 2nd order centered finite difference approximation to the derivative:

  ``( f(x + h) - f(x - h) )/ (2 * h)``

  \input
    :function_p: ``f(x + h)``
    :function_m: ``f(x - h)``
    :epsilon: ``h``
  RETURNS:
    the finite difference approximation
\endrst*/
OL_CONST_FUNCTION OL_WARN_UNUSED_RESULT double SecondOrderCenteredFiniteDifference(double function_p, double function_m, double epsilon) noexcept {
  return (function_p - function_m)/2.0/epsilon;
}

}  // end unnamed namespace

/*!\rst
  Pings the gradient ``\nabla f`` of a function ``f``, using second order finite differences:

  ``grad_f_approximate = ( f(x + h) - f(x - h) )/ (2 * h)``

  This converges as ``O(h^2)`` (Taylor series expansion) for twice-differentiable functions.
  Thus for ``h_1 != h_2``, we can compute two ``grad_f_approximate`` results and thus two errors, ``e_1, e_2``.
  Then (in exact precision), we would obtain:

  ``log(e_1/e_2) / log(h_1/h_2) >= 2``.

  (> sign because superconvergence can happen.)

  Proof: By taylor-expanding ``grad_f_approximate``, we see that:

    ```e = grad_f_anlytic - grad_f_approximate = O(h^2) = c*h^2 + H.O.T`` (Higher Order Terms).

  Assuming ``H.O.T \approx 0``, then

    ``e_1/e_2 \approx c*h_1^2 / (c*h_2^2) = h_1^2 / h_2^2``,

  and

    ``log(e_1/e_2) \approx log(h_1^2 / h_2^2) = 2 * log(h_1/h_2)``.

  Hence

    ``rate = log(e_1/e_2) / log(h_1/h_2) = 2``, *in exact precision.*

  If ``c = 0`` (or is very small), then H.O.T. matters: we could have ``e = 0*h^2 + c_1*h^3 + H.O.T``.
  And we will compute rates larger than 2 (superconvergence). This is not true in general but it can happen.

  This function verifies that the expected convergence rate is obtained with acceptable accuracy in the
  presence of floating point errors.

  This function knows how to deal with vector-valued ``f()`` which accepts a matrix
  of inputs, ``X``.  Analytic gradients and finite difference approximations
  are computed for each output of ``f()`` with respect to each entry in the input, ``X``.  In particular, this
  function can handle ``f()``:

  ``f_k = f(X_{d,i})``

  computing gradients:

  ``gradf_{d,i,k} = \frac{\partial f_k}{\partial X_{i,d}}``.

  So ``d, i`` index the inputs and ``k`` indexes the outputs.

  The basic structure is as follows::

    for i
      for d
        # Compute function values and analytic/finite difference gradients for the current
        # input. Do this for each h value.
        # Step 1
        for each h = h_1, h_2
          X_p = X; X_p[d][i] += h
          X_m = X; X_m[d][i] -= h

          f_p = f(X_p); # k values in f
          f_m = f(X_m); # k values in f

          # Loop over outputs and compute difference between finite difference and analytic results.
          for k
            grad_f_analytic   = get_grad_f[d][i][k](X) // d,i,k entry of gradient evaluated at X
            grad_f_finitediff = (f_p[k] - f_m[k])/(2*h)
            error[k] = grad_f_analytic - grad_f_finitediff
          endfor
        endfor

        # Step 2
        for k
          check error, convergence
        endfor
      endfor
    endfor

  Hence all checks for a specific output (``k``) wrt to a specific point (``d,i``) happen together.
  All dimensions of a given point are grouped together.
  Keep this in mind when writing PingableMatrixInputVectorOutputInterface subclasses as it can make reading debugging output easier.

  See PingableMatrixInputVectorOutputInterface (and its subclasses) for more information/examples on how ``f`` and ``\nabla f``
  can be structured.

  Note that in some cases, this function will decide to skip some tests or run them
  under more relaxed tolerances.  There are many reasons for this:

  1. When the exact gradient is near 0, finite differencing is simply trying
     to compute ``(x1 - x2) = 0``, which has an infinite condition number.  If our
     method is correct/accurate, we will be able to reasonably closely approximate
     0, but we cannot expect convergence.
  2. Backward stability type error analysis deals with normed bounds; for example,
     see the discussion in ResidualNorm().  These normed estimates bound the error
     on the LARGEST entries. The error in the smaller entries can be much larger.

  However, even when errors could be large, we're trying to compute 0, etc., we do
  not want to completely ignore these scenarios since that could cause us to accept
  completely bogus outputs.  Instead we try to compensate.

  Implementer's Note:
  This is an "expert tool." I originally implemented it to automate my workflow when implementing and testing
  "math as code" for gradients. It is a common testing technique from the world of derivative-based optimization
  and nonlinear solvers.

  Generally, this function is to check at a large number of (random) points in the hopes that most of the points
  avoid ill-conditioning issues. Random points are chosen make the implementer's life easier.

  The typical workflow to implement ``f(x)`` and ``df(x)/dx`` might look like:

  1. Code ``f(x)``
  2. Verify ``f(x)``
  3. Analytically compute df/dx (on paper, with a computer algebra system, etc.)
  4. Check ``df/dx``

     a. at some hand-evaluated points
     b. Ping testing (this function)

  If errors arise, this function will output some information to provide further context on what input/output
  combination failed and how. At the head of this file, define OL_PING_TEST_DEBUG_PRINT to turn on super verbose
  printing, which will provide substantially more detail. This is generally useful to see if failing points are just
  barely on the wrong side of tolerance or if there is a larger problem afoot.

  As it turns out, testing derivative code is no easy undertaking. It is often not a well-conditioned problem, and
  the conditioning is difficult to predict, being a function of the input point, function values, and gradient.
  The exact value is knowable from the analytic gradient, but here we are trying to ascertain whether the
  implementation of the analytic gradient is correct!

  Since we do nothing to guarantee good conditioning (random inputs), sometimes our luck is bad. Thus, this function
  includes a number of heuristics to detect conditions in which numerical error is too high to make an informed decision
  on whether we are looking at noise or a genuine error.

  These heuristics involve a number of "magic numbers" defined in the code. These depend heavily
  on the complexity of the gradient being tested (as well as the points at which gradients are being computed).
  Here, we are roughly assuming that the entries of "points" are in [1.0e-3, 1.0e1].
  The magic numbers here are not perfect, but they appear to work in the current use cases. I have left them
  hard-coded for now in the interest of making ping test easier to set up. So, apologies in advance for the
  magic numbers.

  We chose the bypasses to heavily favor eliminating false positives. Some true positives are *also lost*.
  This is why OL_PING_TEST_DEBUG_PRINTF is important when debugging. If a true positive comes up, then
  there are probably many more, which can be viewed by parsing the more detailed output.

  .. WARNING:: This function is NOT a catch-all. With the heuristics in place, you could adversarially construct an
    implementation that is wrong but passes this test.

  .. WARNING:: Additionally, this function cannot detect errors that are not in the gradient. If you intended
    to implement ``f = sin(x)``, ``f' = cos(x)``, but instead coded ``g = sin(x) + 1``, this function will not find it.
    This cannot detect small amounts of noise in the gradient either (e.g., ``f' = cos(x) + 1.0e-15``). There is no
    way to tell whether that is noise due to numerical error or noise due to incorrectness.

  TODO(GH-162): thresholds are an imperfect tool for this task. Loss of precision is not a binary event; you are not
  certain at ``2^{-52}`` but uncertain at ``2^{-51}``. It might be better to estimate the range over which we go from
  meaningful loss of precision to complete noise and have a linear ramp for the tolerance over that space. Maybe
  it should be done in log-space?
\endrst*/
int PingDerivative(const PingableMatrixInputVectorOutputInterface& function_and_derivative_evaluator, double const * restrict points, double epsilon[2], double rate_tolerance_fine, double rate_tolerance_relaxed, double input_output_ratio) noexcept {
  int num_rows, num_cols;
  function_and_derivative_evaluator.GetInputSizes(&num_rows, &num_cols);
  int num_outputs = function_and_derivative_evaluator.GetOutputSize();

  // *_p means "plus" and refers either to x+epsilon or f(x+epsilon)
  // *_m means "minus"
  std::vector<double> function_p(2*num_outputs);
  std::vector<double> function_m(2*num_outputs);
  std::vector<double> error(2*num_outputs);
  std::vector<double> points_p(num_rows*num_cols);
  std::vector<double> points_m(num_rows*num_cols);

  const double rate_exact = 2.0;
  // since the floating point representation of x+h is not exactly x+h, we want to compensate for the difference
  // see below for details on how epsilon_actual is computed
  double epsilon_actual[2];

  int ping_failures = 0;
  for (int i_cols = 0; i_cols < num_cols; ++i_cols) {
    for (int i_rows = 0; i_rows < num_rows; ++i_rows) {  // i, i_rows index over what we're differentiating against
      // Now we have a single i_cols and i_rows, specifying a single input out of n_cols * n_rows total.
      // Think of this as "x" (= X[i_cols][i_rows]).

      // Step 1: compute function values and gradients at/around "x", for both values of epsilon.
      for (int i_epsilon = 0; i_epsilon < 2; ++i_epsilon) {  // i_epsilon over h choices
        // Reset X_p and X_m to X. We only want to vary ONE index at a time, the variable x.
        for (int j = 0; j < num_rows*num_cols; ++j) {
          points_p[j] = points[j];
          points_m[j] = points[j];
        }

        // The input epsilon[2] is *not* exactly the h values used in computing f(x+h), f(x-h). Depending on the value
        // of x, the floating point value (denoted fl()) fl(x+h) = x + h + e != x + h.
        // An extreme example: x = 1000.0, h = 1.0e-16, then fl(x+h) = x exactly in double precision and e = -h.
        // To compensate, we'd like to estimate e.
        // So we compute the h', epsilon_actual. h' = fl(-x + fl(x + h)) = fl(-x + x + h + e) = fl(h + e) = h + e + e'.
        // As long as h << x (this function makes no sense otherwise), then e' \approx h * machine_epsilon and our h'
        // is good enough.

        // Initialize to -points_p, which is currently just a copy of points.
        // Note: we only compute one set of epsilon_actual per (i_cols, i_rows) so there is no need
        // to store any additional terms.
        epsilon_actual[i_epsilon] = -points_p[i_cols*num_rows + i_rows];

        points_p[i_cols*num_rows + i_rows] += epsilon[i_epsilon];  // x_p = x + h
        points_m[i_cols*num_rows + i_rows] -= epsilon[i_epsilon];  // x_m = x - h

        // Compute the actual h: h' = fl(-x + fl(x + h)) = fl(h + e)
        epsilon_actual[i_epsilon] += points_p[i_cols*num_rows + i_rows];

        // Compute f_p = f(X_p), f_m = f(X_m)
        function_and_derivative_evaluator.EvaluateFunction(points_p.data(), function_p.data() + i_epsilon*num_outputs);
        function_and_derivative_evaluator.EvaluateFunction(points_m.data(), function_m.data() + i_epsilon*num_outputs);

        // For each output, compute the difference (error) between analytic gradient and finite difference gradient.
        // Note: we save the error values for all outputs and for both epsilon values; we need the full set for
        // diagnostics later.
        for (int k = 0; k < num_outputs; ++k) {
          double grad_function_f = SecondOrderCenteredFiniteDifference(function_p[i_epsilon*num_outputs + k], function_m[i_epsilon*num_outputs + k], epsilon_actual[i_epsilon]);
          double grad_function_a = function_and_derivative_evaluator.GetAnalyticGradient(i_rows, i_cols, k);

          error[k*2 + i_epsilon] = std::fabs(grad_function_a - grad_function_f);
        }  // end for k: num_outputs
      }  // end for i_epsilon: 2

      // Step 2: check for errors, convergence, etc. in the data from step 1.
      // Since we used second differences to approximate the derivatve, we expect second order convergence.
      // test that this happens.
      // NOTE: when the analytic derivative is 'small' (near 0), this is a VERY poorly conditioned problem.  It's akin
      // to trying to compute (x1 - x2) = 0, so we need a lot of logic to realize when the problem is poorly conditioned
      // and bypass the convergence rate check.
      for (int k = 0; k < num_outputs; ++k) {
        // Get a rough idea of how large the function (k-th output value) is at X_p, X_m. We don't use f_k(X) in case the
        // function is changing very rapidly. We want a measure the magnitudes that went into the f(X_p) - f(X_m) computation.
        const double function_value_norm = std::sqrt(Square(function_p[num_outputs + k]) + Square(function_m[num_outputs + k])) + std::numeric_limits<double>::min();  // if both values are 0, adding ~1e-308 prevents division by 0 without negatively affecting any dependent computations
        const double grad_function_a = function_and_derivative_evaluator.GetAnalyticGradient(i_rows, i_cols, k);
        const double fabs_grad_function_a = std::fabs(grad_function_a);

        // The following code performs a series of checks on the relative sizes of the error, function values, gradient values
        // and input values. We are attempting to identify cases where we do not have enough precision to compute the
        // convergence rate accurately. When this happens, a "continue" statement will execute and we will skip this
        // particular combination of input point and output value.

        // At the end, if none of the "continue" statements are hit, then we are confident that the computed convergence
        // rate is accurate, and we will check that it is within tolerance of the expected convergence rate.
        // In general, we want to eliminate false positives so that errors genuinely indicate something broken in the code.

        // First, we do checks based on the magnitude of the error relative to the function value (f) and input value (X).

        // Result is extremely accurate: so much so that the rate of convergence
        // cannot be accurately calculated. 1.0e-20 was chosen to be a little smaller than machine precision (~1.0e-16).
        // For ||X|| \approx 1, we don't have enough precision to examine f(X_p) - f(X_m) at the 20th decimal place
        // and beyond, which we need to do in order to compute the convergence rate.
        if (error[k*2+0]/function_value_norm < 1.0e-20 && error[k*2+1]/function_value_norm < 1.0e-20) {
          continue;
        }

        // Like the previous check, but now comparing against magnitude of X[i_cols][i_rows], the value that was tweaked
        // by epsilon to produce X_p and X_m. The same logic applies about lacking sufficient precision to examine f(X_p) - f(X_m).
        if (error[k*2+0]/std::fabs(points[i_cols*num_rows + i_rows]) < input_output_ratio && error[k*2+1]/std::fabs(points[i_cols*num_rows + i_rows]) < input_output_ratio) {
          continue;
        }

        // Compute the convergence rate and the absolute error against the expected convergence rate.
        const double rate = std::log10(error[k*2 + 0]/error[k*2 + 1]) / std::log10(epsilon_actual[0]/epsilon_actual[1]);
        const double rate_error = std::fabs(rate_exact - rate);
        OL_PING_TEST_DEBUG_PRINTF("k=%d, rate = %.18E, |2 - rate| = %.18E\n", k, rate, rate_error);

        // As mentioned in the function comments, loss of precision is not a binary event. This is an attempt to "ramp up"
        // the tolerable loss over a range: for larger values of fabs_grad_function_a/function_value_norm, we expect to be
        // safe. For small values (e.g., fabs_grad_function_a near 0), we expect worse conditioning.
        // So we give a linear ramp to increase tolerance. (Maybe this should be done in log10 space.)
        // 1.0e-2 was chosen empirically.
        double rate_tolerance = rate_tolerance_fine;
        if (fabs_grad_function_a/function_value_norm < 1.0e-2) {
          rate_tolerance = rate_tolerance_relaxed + (fabs_grad_function_a/function_value_norm)/1.0e-2*(rate_tolerance_fine - rate_tolerance_relaxed);
          OL_PING_TEST_DEBUG_PRINTF("rate_tolerance = %.18E\n", rate_tolerance);
        }

        // Now we will do numerical precision checks related to the convergence rate.

        OL_PING_TEST_DEBUG_PRINTF("Checking convergence rate...\n");
        // isnan(rate) == true probably means epsilon[0] == epsilon[1]; then rate was computed as log(1)/log(1) = 0/0, NaN.
        // NaN can also arise if error[k*2+0] == error[k*2+1] == 0, BUT we would have already skipped if this happens
        if (rate_error > rate_tolerance || std::isnan(rate)) {
          // rate == 0.0, implying error[k*2+0] == error[k*2+1] != 0, should never happen. Log something extra when it does.
          if (rate == 0.0) {
            OL_PING_TEST_DEBUG_PRINTF("RATE == 0.0! error[%d*2 + 0] = %.18E, error[%d*2 + 1] = %.18E\n", k, error[k*2+0], k, error[k*2+1]);
          }

          // (Substantially) better than expected convergence is ok
          if (rate > 1.2*rate_exact && rate < 3*rate_exact) {
            continue;
          }

          // We are estimating grad_function_a by differencing two quantities with magnitude \approx function_value_norm.
          // If the ratio between these is small, then there is not enough precision in f(X_p) - f(X_m) to compute
          // the convergence rate. When this happens, it is ok if error[k*2+0] \approx error[k*2+1] and we are
          // satisfied if the relative error is small.
          OL_PING_TEST_DEBUG_PRINTF("analytic/value: %.18E, error/value: %.18E\n", fabs_grad_function_a/function_value_norm, error[k*2+1]/function_value_norm);
          if (fabs_grad_function_a/function_value_norm < 1.0e-5 && error[k*2+1]/function_value_norm < 1.0e-11) {
            continue;
          }

          // Given our assumption about the coordinate-wise magnitude of the input points, if function_value_norm is
          // small, then we lost substantial precision just computing f(X_p) and f(X_m). Intuitively, if I try to
          // compute 1.0e-8 as x - y where |x|, |y| \approx 1, then I have lost half the digits in x and y.
          if (function_value_norm < 1.0e-8 && error[k*2+1]/function_value_norm < 1.0e-12) {
            continue;
          }

          // All numerical precision bypasses have passed. We are confident that we computed the convergence rate
          // accurately AND that it is wrong.
          ++ping_failures;

          // Print information about the indexes, point, function, gradient (analytic and finite diff) and error
          // at the X[i_cols][i_rows], f'(X) combination that we judged was computed incorrectly.
          OL_ERROR_PRINTF("point[%d,%d] = %.18E\n", i_rows, i_cols, points[i_cols*num_rows + i_rows]);

          for (int i_epsilon = 0; i_epsilon < 2; ++i_epsilon) {
            double grad_function_f = SecondOrderCenteredFiniteDifference(function_p[i_epsilon*num_outputs + k], function_m[i_epsilon*num_outputs + k], epsilon_actual[i_epsilon]);
            OL_ERROR_PRINTF("i_rows = %d, i_cols = %d, k = %d, epsilon = %.6E\n", i_rows, i_cols, k, epsilon_actual[i_epsilon]);
            OL_ERROR_PRINTF("fcn_p[k] = %.18E, fcn_m[k] = %.18E\n", function_p[i_epsilon*num_outputs + k], function_m[i_epsilon*num_outputs + k]);
            OL_ERROR_PRINTF("analytic: %.18E\n", grad_function_a);
            OL_ERROR_PRINTF("finite  : %.18E\n", grad_function_f);
            OL_ERROR_PRINTF("diff    : %.18E\n", std::fabs(grad_function_a - grad_function_f));  // = error[2*k + i_epsilon]
          }

          OL_ERROR_PRINTF("ERROR PING FAILED on i_rows = %d, i_cols = %d, k = %d\n", i_rows, i_cols, k);
          OL_ERROR_PRINTF("k=%d, rate = %.18E, |2 - rate| = %.18E\n", k, rate, rate_error);
        }
      }  // end for k: num_outputs
    }  // end for i_rows: num_rows
  }  // end for i_cols: num_cols

  return ping_failures;
}

void ExpandDomainBounds(double scale_factor, std::vector<ClosedInterval> * domain_bounds) {
#ifdef OL_VERBOSE_PRINT
  PrintDomainBounds(domain_bounds.data(), dim);  // domain before expansion
#endif
  for (auto& interval : *domain_bounds) {
    double side_length = interval.Length();
    double midpoint = 0.5*(interval.max + interval.min);
    side_length *= scale_factor;
    side_length *= 0.5;  // want half of side_length b/c we grow outward from the midpoint
    interval = {midpoint - side_length, midpoint + side_length};
  }
#ifdef OL_VERBOSE_PRINT
  PrintDomainBounds(domain_bounds.data(), dim);  // expanded domain
#endif
}

void FillRandomCovarianceHyperparameters(const boost::uniform_real<double>& uniform_double_hyperparameter, UniformRandomGenerator * uniform_generator, std::vector<double> * hyperparameters, CovarianceInterface * covariance) {
  std::generate(hyperparameters->begin(), hyperparameters->end(), [&uniform_double_hyperparameter, uniform_generator]() {
    return uniform_double_hyperparameter(uniform_generator->engine);
  });
  covariance->SetHyperparameters(hyperparameters->data());
}

void FillRandomDomainBounds(const boost::uniform_real<double>& uniform_double_lower_bound, const boost::uniform_real<double>& uniform_double_upper_bound, UniformRandomGenerator * uniform_generator, std::vector<ClosedInterval> * domain_bounds) {
  std::generate(domain_bounds->begin(), domain_bounds->end(), [&uniform_double_lower_bound, &uniform_double_upper_bound, uniform_generator]() {
    double min = uniform_double_lower_bound(uniform_generator->engine);
    double max = uniform_double_upper_bound(uniform_generator->engine);
    return ClosedInterval(min, max);
  });
}

void FillRandomGaussianProcess(double const * restrict points_to_sample, double const * restrict noise_variance, int dim, int num_to_sample, double * restrict points_to_sample_value, GaussianProcess * gaussian_process) {
  for (int j = 0; j < num_to_sample; ++j) {
    // draw function value from the GP
    points_to_sample_value[j] = gaussian_process->SamplePointFromGP(points_to_sample, noise_variance[j]);
    // add function value back into the GP
    gaussian_process->AddPointsToGP(points_to_sample, points_to_sample_value + j, noise_variance + j, 1);
    points_to_sample += dim;
  }
}

}  // end namespace optimal_learning
