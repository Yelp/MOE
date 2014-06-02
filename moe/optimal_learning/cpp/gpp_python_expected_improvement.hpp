/*!
  \file gpp_python_expected_improvement.hpp
  \rst
  This file registers the translation layer for invoking ExpectedImprovement functions
  (e.g., computing/optimizing EI; see gpp_math.hpp) from Python.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Exports subclasses of ObjectiveEstimationPolicyInterface (e.g., ConstantLiar, KrigingBeliever).
  These classes do not have any useful behavior inside Python; instead (like GaussianProcess), they
  are meant to be constructed and passed back to C++.
\endrst*/
void ExportEstimationPolicies();

/*!\rst
  Exports functions (with docstrings) for expected improvement operations:

  1. expected improvement (and its gradient) evaluation (uesful for testing)
  2. multistart expected improvement optimization (main entry-point)
  3. expected improvement evaluation at a list of points (useful for testing, plotting)

  These functions choose between monte-carlo and analytic EI evaluation automatically.
\endrst*/
void ExportExpectedImprovementFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_EXPECTED_IMPROVEMENT_HPP_
