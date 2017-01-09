/*!
  \file gpp_model_selection_test.hpp
  \rst
  Functions for testing gpp_model_selection's functionality--the evaluation of
  LogMarginalLikelihood and LeaveOneOutLogLikelihood (plus gradient, hessian) and the optimization of these
  metrics wrt hyperparameters of the covariance function.

  These will be abbreviated as:

  * LML = LogMarginalLikelihood
  * LOO-CV = Leave One Out Cross Validation

  As in gpp_math_test, we have two main groups of tests:

  * ping (unit) tests for gradient/hessian of LML and gradient of LOO-CV.
  * unit + integration tests for optimization methods (gradient descent, newton)

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

  Finally, we have integration tests for LML and LOO-CV optimization.  Unit tests for optimizers live in
  gpp_optimization_test.hpp/cpp.  These integration tests use constructed data but exercise all the
  same code paths used for hyperparameter optimization in production.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_TEST_HPP_

#include "gpp_common.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

/*!\rst
  Runs a battery of ping tests for the Log Likelihood Evaluators:

  * Log Marginal: gradient and hessian wrt hyperparameters
  * Leave One Out: gradient wrt hyperparameters

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunLogLikelihoodPingTests();

/*!\rst
  Checks that hyperparameter optimization is working for the selected combination of
  OptimizerTypes (gradient descent, newton) and LogLikelihoodTypes (log marginal
  likelihood, leave-one-out cross-validation log pseudo-likelihood).

  .. Note:: newton and leave-one-out is not implemented.

  \param
    :optimizer_type: which optimizer to use
    :objective_mode: which log likelihood measure to use
  \return
    number of test failures: 0 if hyperparameter optimization (based on marginal likelihood) is working properly
\endrst*/
OL_WARN_UNUSED_RESULT int HyperparameterLikelihoodOptimizationTest(OptimizerTypes optimizer_type, LogLikelihoodTypes objective_mode);

/*!\rst
  Tests EvaluateLogLikelihoodAtPointList (computes log likelihood at a specified list of hyperparameters, multithreaded).
  Checks that the returned best point is in fact the best.
  Verifies multithreaded consistency.

  \return
    number of test failures: 0 if function evaluation is working properly
\endrst*/
OL_WARN_UNUSED_RESULT int EvaluateLogLikelihoodAtPointListTest();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_TEST_HPP_
