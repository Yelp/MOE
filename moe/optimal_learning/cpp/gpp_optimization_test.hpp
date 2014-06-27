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
  Checks that the following optimizers are working correctly with simple objectives:

  * kGradientDescent
  * kNewton

  by checking unconstrained and constrained optimization against polynomial
  objective function(s).

  Also checks that MultistartOptimizer::MultistartOptimize() handles exceptions correctly and without crashing.

  \return
    number of test failures: 0 if optimizer is working properly
\endrst*/
int RunOptimizationTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_TEST_HPP_
