// gpp_model_selection_and_hyperparameter_optimization_test.hpp
/*
  Functions for testing gpp_model_selection_and_hyperparameter_optimization's functionality--the evaluation of
  LogMarginalLikelihood and LeaveOneOutLogLikelihood (plus gradient, hessian) and the optimization of these
  metrics wrt hyperparameters of the covariance function.

  These will be LML = LogMarginalLikelihood, LOO-CV = Leave One Out Cross Validation for short.

  As in gpp_math_test, we have two main groups of tests:
  <> ping (unit) tests for gradient/hessian of LML and gradient of LOO-CV.
  <> unit + integration tests for optimization methods (gradient descent, newton)

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

  Finally, we have integration tests for LML and LOO-CV optimization.  Unit tests for optimizers live in
  gpp_optimization_test.hpp/cpp.  These integration tests use constructed data but exercise all the
  same code paths used for hyperparameter optimization in production.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MODEL_SELECTION_AND_HYPERPARAMETER_OPTIMIZATION_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MODEL_SELECTION_AND_HYPERPARAMETER_OPTIMIZATION_TEST_HPP_

#include "gpp_common.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization.hpp"
#include "gpp_optimization.hpp"

namespace optimal_learning {

/*
  Runs a battery of ping tests for the Log Likelihood Evaluators:
  <> Log Marginal: gradient and hessian wrt hyperparameters
  <> Leave One Out: gradient wrt hyperparameters

  RETURNS:
  number of test failures: 0 if all is working well.
*/
OL_WARN_UNUSED_RESULT int RunLogLikelihoodPingTests();

/*
  Checks that hyperparameter optimization is working for the selected combination of
  OptimizerTypes (gradient descent, newton) and LogLikelihoodTypes (log marginal
  likelihood, leave-one-out cross-validation log pseudo-likelihood).

  Note: newton and leave-one-out is not implemented.

  INPUTS:
  optimizer_type: which optimizer to use
  objective_mode: which log likelihood measure to use
  RETURNS:
  number of test failures: 0 if hyperparameter optimization (based on marginal likelihood) is working properly
*/
OL_WARN_UNUSED_RESULT int HyperparameterLikelihoodOptimizationTest(OptimizerTypes optimizer_type, LogLikelihoodTypes objective_mode);

/*
  Tests EvaluateLogLikelihoodAtPointList (computes log likelihood at a specified list of hyperparameters, multithreaded).
  Checks that the returned best point is in fact the best.
  Verifies multithreaded consistency

  RETURNS:
  number of test failures: 0 if function evaluation is working properly
*/
OL_WARN_UNUSED_RESULT int EvaluateLogLikelihoodAtPointListTest();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MODEL_SELECTION_AND_HYPERPARAMETER_OPTIMIZATION_TEST_HPP_
