// gpp_python_expected_improvement.hpp
/*
  This file registers the translation layer for invoking ExpectedImprovement functions
  (e.g., computing/optimizing EI; see gpp_math.hpp) from Python.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Exports subclasses of ObjectiveEstimationPolicyInterface (e.g., ConstantLiar, KrigingBeliever).
  These classes do not have any useful behavior inside Python; instead (like GaussianProcess), they
  are meant to be constructed and passed back to C++.
*/
void ExportEstimationPolicies();

/*
  Exports functions (with docstrings) for expected improvement operations:
  1) expected improvement (and its gradient) evaluation (uesful for testing)
  2) multistart expected improvement optimization (main entry-point)
  3) expected improvement evaluation at a list of points (useful for testing, plotting)
  These functions choose between monte-carlo and analytic EI evaluation automatically.
*/
void ExportExpectedImprovementFunctions();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_
