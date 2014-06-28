/*!
  \file gpp_python_model_selection.hpp
  \rst
  This file registers the translation layer for invoking functions in
  gpp_model_selection.hpp from Python.

  The functions exported expect hyperparameters input like "pylist hyperparameters[2]:"
  This will be a python list such that:
  ``hyperparameters[0] = double`` precision number: ``\alpha`` (``=\sigma_f^2``, signal variance)
  ``hyperparameters[1] = pylist lengths[dim]``: list of length scales for covariance (doubles)
  For example, if dim = 3, you might set in Python:
    ``hyperparameters_for_C = [2.1, [1.2, 3.1, 0.4]]``
  for ``\alpha = 2.1``, and for length scales 1.2, 3.1, 0.4
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Exports functions (with docstrings) for model selection:

  1. log likelihood (and its gradient) evaluation (useful for testing)
  2. multistart hyperparameter optimization (main entry-point)
  3. log likelihood evaluation at a list of hyperparameters (useful for testing, plotting)
\endrst*/
void ExportModelSelectionFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_MODEL_SELECTION_HPP_
