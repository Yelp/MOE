/*!
  \file gpp_python_gaussian_process.hpp
  \rst
  This file registers the translation layer for constructing a GaussianProcess
  and invoking its member functions (see gpp_math.hpp) from Python.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Exports constructor and member functions (with docstrings) from GaussianProcess:

  1. Constructor accepting Python structures
  2. Evaluation of mean, variance, cholesky of variance (and their gradients)
\endrst*/
void ExportGaussianProcessFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_GAUSSIAN_PROCESS_HPP_
