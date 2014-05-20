/*!
  \file gpp_optimization_test.hpp
  \rst
  Functions to test the optimization algorithms in gpp_optimization.hpp.

  Calls optimizers with simple analytic functions (e.g., polynomials) with known optima
  and verifies that the optimizers can find the solutions.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_TEST_HPP_

#include "gpp_common.hpp"
#include "gpp_optimization.hpp"

namespace optimal_learning {

/*!\rst
  Checks that specified optimizer is working correctly:
  * kGradientDescent
  * kNewton

  Checks unconstrained and constrained optimization against polynomial
  objective function(s).

  \param
    :optimizer_type: which optimizer to test
  \return
    number of test failures: 0 if optimizer is working properly
\endrst*/
int RunOptimizationTests(OptimizerTypes optimizer_type);

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_TEST_HPP_
