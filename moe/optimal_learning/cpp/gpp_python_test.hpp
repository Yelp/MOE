/*!
  \file gpp_python_test.hpp
  \rst
  This file registers a Python function that invokes all of the C++ tests.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Runs all C++ unit tests.
\endrst*/
void ExportCppTestFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TEST_HPP_
