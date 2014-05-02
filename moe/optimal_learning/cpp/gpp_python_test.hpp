// gpp_python_test.hpp
/*
  This file registers a Python function that invokes all of the C++ tests.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Runs all C++ unit tests.
*/
void ExportCppTestFunctions();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_TEST_HPP_
