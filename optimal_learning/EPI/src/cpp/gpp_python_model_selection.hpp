// gpp_python_model_selection.hpp
/*
  This file registers the translation layer for invoking functions in
  gpp_model_selection_and_hyperparameter_optimization.hpp from Python.

  The functions exported expect hyperparameters input like "pylist hyperparameters[2]:"
  This will be a python list such that:
  hyperparameters[0] = double precision number: \alpha (=\sigma_f^2, signal variance)
  hyperparameters[1] = pylist lengths[dim]: list of length scales for covariance (doubles)
  For example, if dim = 3, you might set in Python:
    hyperparameters_for_C = [2.1, [1.2, 3.1, 0.4]]
  for \alpha = 2.1, and for length scales 1.2, 3.1, 0.4
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Exports functions (with docstrings) for model selection:
  1) log likelihood (and its gradient) evaluation (useful for testing)
  2) multistart hyperparameter optimization (main entry-point)
  3) log likelihood evaluation at a list of hyperparameters (useful for testing, plotting)
*/
void ExportModelSelectionFunctions();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_
