/*!
  \file gpp_math_test.hpp
  \rst
  Functions for testing gpp_math's GP and EI functionality.

  Tests are broken into two main groups:

  * ping (unit) tests for GP outputs (mean, cholesky/variance) and EI (for the general and one sample cases)
  * unit + integration tests for optimization methods

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

  There is also a consistency check between general MC-based EI calculation and the analytic one sample case.

  Finally, we have tests for EI optimization.  These include multithreading tests (verifying that each core
  does what is expected) as well as integration tests for EI optimization.  Unit tests for optimizers live in
  gpp_optimization_test.hpp/cpp.  These integration tests use constructed data but exercise all the
  same code paths used for hyperparameter optimization in production.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_TEST_HPP_

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_random.hpp"
#include "gpp_exception.hpp"
#include "gpp_math.hpp"

namespace optimal_learning {

/*!\rst
  Enum for specifying which EI evaluation mode to test.
\endrst*/
enum class ExpectedImprovementEvaluationMode {
  //! test analytic evaluation
  kAnalytic = 0,
  //! test monte-carlo evaluation
  kMonteCarlo = 1,
};

/*!\rst
  Checks that the gradients (spatial) of the GP mean are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingGPMeanTest();

/*!\rst
  Checks that the gradients (spatial) of the GP variance are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingGPVarianceTest();

/*!\rst
  Checks that the gradients (spatial) of the cholesky factorization of GP variance are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingGPCholeskyVarianceTest();

/*!\rst
  Supports evaluating an analytic special case of expected improvement via OnePotentialSampleExpectedImprovementEvaluator.

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}`` (with i always indexing 0).
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
\endrst*/
class PingOnePotentialSampleExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI ONE potential sample analytic";

  PingOnePotentialSampleExpectedImprovement(double const * restrict lengths, double const * restrict OL_UNUSED(points_being_sampled), double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int OL_UNUSED(num_to_sample), int num_being_sampled, int num_sampled, int OL_UNUSED(num_mc_iter)) OL_NONNULL_POINTERS;

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS; 

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT; 

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT; 

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2); 

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT; 

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS; 

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI). Must be 0 for the analytic case.
  const static int num_being_sampled_ = 0;
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! the gradient of EI at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_EI_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for EI evaluation
  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingOnePotentialSampleExpectedImprovement);
};

/*!\rst
  Supports evaluating the expected improvement, ExpectedImprovementEvaluator::ComputeExpectedImprovement() and
  its gradient, ExpectedImprovementEvaluator::ComputeGradExpectedImprovement()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
\endrst*/
class PingExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI with MC integration";

  PingExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled, int num_sampled, int num_mc_iter) OL_NONNULL_POINTERS;

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS; 

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT; 

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT; 

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2); 

  virtual double GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT; 

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS; 

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  int num_to_sample_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  int num_being_sampled_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! points that are being sampled in concurrently experiments
  std::vector<double> points_being_sampled_;
  //! the gradient of EI at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_EI_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for EI evaluation
  ExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingExpectedImprovement);
};

/*!\rst
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases
  Works with various EI evaluators (e.g., MC, analytic formulae)

  \param
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping/test failures
\endrst*/
template <typename EIEvaluator>
OL_WARN_UNUSED_RESULT int PingEITest(int num_to_sample, int num_being_sampled, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 7.0;
  const int num_mc_iter = 16;

  MockExpectedImprovementEnvironment EI_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    EIEvaluator EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
    EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);
    errors_this_iteration = PingDerivative(EI_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s (%d,%d-EI) gradient pings failed with %d errors\n", EIEvaluator::kName, num_to_sample, num_being_sampled, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s (%d,%d-EI) gradient pings passed\n", EIEvaluator::kName, num_to_sample, num_being_sampled);
  }

  return total_errors;
}

/*!\rst
  Checks that the gradients (spatial) of Expected Improvement are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingEIGeneralTest();

/*!\rst
  Checks the gradients (spatial) of Expected Improvement (in the special case of only 1 potential sample) are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingEIOnePotentialSampleTest();

/*!\rst
  Runs a battery of ping tests for the GP and optimization functions:

  * GP mean
  * GP variance
  * cholesky decomposition of the GP variance
  * Expected Improvement
  * Expected Improvement special case: only *ONE* potential point to sample

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunGPPingTests();

/*!\rst
  Tests that the general EI + grad EI computation (using MC integration) is consistent
  with the special analytic case of EI when there is only *ONE* potential point
  to sample.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunEIConsistencyTests();

/*!\rst
  Checks that multithreaded EI optimization behaves the same way that single threaded does.

  \param
    :ei_mode: ei evaluation mode to test (analytic or monte carlo)
  \return
    number of test failures: 0 if EI multi/single threaded optimization are consistent
\endrst*/
OL_WARN_UNUSED_RESULT int MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode ei_mode);

/*!\rst
  Checks that EI optimization is working on tensor product or simplex domain using
  analytic or monte-carlo EI evaluation.

  \param
    :domain_type: type of the domain to test on (e.g., tensor product, simplex)
    :ei_mode: ei evaluation mode to test (analytic or monte carlo)
  \return
    number of test failures: 0 if EI optimization is working properly
\endrst*/
OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationTest(DomainTypes domain_type, ExpectedImprovementEvaluationMode ei_mode);

/*!\rst
  Checks that ComputeOptimalPointsToSample works on a tensor product domain.
  This test exercises the the code tested in:
  ExpectedImprovementOptimizationTest(kTensorProduct, ei_mode)
  for ``ei_mode = {kAnalytic, kMonteCarlo}``.

  This test checks the generation of multiple, simultaneous experimental points to sample.

  \return
    number of test failures: 0 if EI optimization is working properly
\endrst*/
OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationMultipleSamplesTest();

/*!\rst
  Tests EvaluateEIAtPointList (computes EI at a specified list of points, multithreaded).
  Checks that the returned best point is in fact the best.
  Verifies multithreaded consistency.

  \return
    number of test failures: 0 if function evaluation is working properly
\endrst*/
OL_WARN_UNUSED_RESULT int EvaluateEIAtPointListTest();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_TEST_HPP_
