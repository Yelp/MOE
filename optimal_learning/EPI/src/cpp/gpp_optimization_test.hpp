// gpp_optimization_test.hpp
/*
  Functions to test the optimization algorithms in gpp_optimization.hpp.

  Calls optimizers with simple analytic functions (e.g., polynomials) with known optima
  and verifies that the optimizers can find the solutions.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_TEST_HPP_

#include "gpp_common.hpp"
#include "gpp_optimization.hpp"

namespace optimal_learning {

/*
  Checks that specified optimizer is working correctly:
    kGradientDescent
    kNewton
  Checks unconstrained and constrained optimization against polynomial
  objective function(s).

  INPUTS:
  optimizer_type: which optimizer to test
  RETURNS:
  number of test failures: 0 if optimizer is working properly
*/
int RunOptimizationTests(OptimizerTypes optimizer_type);

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_TEST_HPP_
