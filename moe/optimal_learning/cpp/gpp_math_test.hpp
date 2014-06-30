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

#include "gpp_common.hpp"
#include "gpp_domain.hpp"

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
  Runs a battery of tests for the GP and EI functions, including ping tests for:

  * GP mean
  * GP variance
  * cholesky decomposition of the GP variance
  * Expected Improvement
  * Expected Improvement special case: only *ONE* potential point to sample

  and edge case testing for:

  * 1D Analytic Expected Improvement

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunGPTests();

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
