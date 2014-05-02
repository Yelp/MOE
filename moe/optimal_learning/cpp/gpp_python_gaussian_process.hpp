// gpp_python_gaussian_process.hpp
/*
  This file registers the translation layer for constructing a GaussianProcess
  and invoking its member functions (see gpp_math.hpp) from Python.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Exports constructor and member functions (with docstrings) from GaussianProcess:
  1) constructor accepting Python structures
  2) evaluation of mean, variance, cholesky of variance (and their gradients)
*/
void ExportGaussianProcessFunctions();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_
