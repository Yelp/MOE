// gpp_python_common.cpp
/*
  This file contains definitions of infrequently-used or expensive functions declared in gpp_python_common.hpp.

  Note: several internal functions of this source file are only called from Export*() functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  Export*() as Python docstrings, so we saw no need to repeat ourselves.
*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_common.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <cstdio>  // NOLINT(build/include_order)

#include <vector>  // NOLINT(build/include_order)

#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/enum.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

void CopyPylistToVector(const boost::python::list& input, int size, std::vector<double>& output) {
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    output[i] = boost::python::extract<double>(input[i]);
  }
}

void CopyPylistToClosedIntervalVector(const boost::python::list& input, int size, std::vector<ClosedInterval>& output) {
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    output[i].min = boost::python::extract<double>(input[2*i + 0]);
    output[i].max = boost::python::extract<double>(input[2*i + 1]);
  }
}

boost::python::list VectorToPylist(const std::vector<double>& input) {
  boost::python::list result;
  for (const auto& entry : input) {
    result.append(entry);
  }
  return result;
}

PythonInterfaceInputContainer::PythonInterfaceInputContainer(const boost::python::list& points_to_sample_in, int dim_in, int num_to_sample_in)
    : dim(dim_in),
      num_to_sample(num_to_sample_in),
      points_to_sample(dim*num_to_sample) {
  CopyPylistToVector(points_to_sample_in, dim*num_to_sample, points_to_sample);
}

PythonInterfaceInputContainer::PythonInterfaceInputContainer(const boost::python::list& points_to_sample_in, const boost::python::list& points_being_sampled_in, int dim_in, int num_to_sample_in, int num_being_sampled_in)
    : dim(dim_in),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      points_to_sample(dim*num_to_sample),
      points_being_sampled(dim*num_being_sampled) {
  CopyPylistToVector(points_to_sample_in, dim*num_to_sample, points_to_sample);
  CopyPylistToVector(points_being_sampled_in, dim*num_being_sampled, points_being_sampled);
}

PythonInterfaceInputContainer::PythonInterfaceInputContainer(const boost::python::list& hyperparameters_in, const boost::python::list& points_sampled_in, const boost::python::list& points_sampled_value_in, const boost::python::list& noise_variance_in, const boost::python::list& points_to_sample_in, int dim_in, int num_sampled_in, int num_to_sample_in)
    : dim(dim_in),
      num_sampled(num_sampled_in),
      num_to_sample(num_to_sample_in),
      alpha(boost::python::extract<double>(hyperparameters_in[0])),
      lengths(dim),
      points_sampled(dim*num_sampled),
      points_sampled_value(num_sampled),
      noise_variance(num_sampled),
      points_to_sample(0) {
  const boost::python::list& lengths_in = boost::python::extract<boost::python::list>(hyperparameters_in[1]);
  CopyPylistToVector(lengths_in, dim, lengths);
  CopyPylistToVector(points_sampled_in, dim*num_sampled, points_sampled);
  CopyPylistToVector(points_sampled_value_in, num_sampled, points_sampled_value);
  CopyPylistToVector(noise_variance_in, num_sampled, noise_variance);
  CopyPylistToVector(points_to_sample_in, dim*num_to_sample, points_to_sample);
}

int RandomnessSourceContainer::SetNormalRNGSeedPythonList(const boost::python::list& seed_list, const boost::python::list& seed_flag_list) {
  auto seed_list_len = boost::python::len(seed_list);
  auto seed_flag_list_len = boost::python::len(seed_flag_list);
  IdentifyType<decltype(seed_flag_list_len)>::type num_threads = normal_rng_vec.size();
  if (unlikely(seed_list_len != seed_flag_list_len || seed_list_len != num_threads)) {
    OL_ERROR_PRINTF("NORMAL RNG SEEDING ERROR: len(seed_list) = %lu, len(seed_flag_list) = %lu, normal_rng_vec.size() = %lu\n", seed_list_len, seed_flag_list_len, num_threads);
    OL_ERROR_PRINTF("Seed list & flag list must be the same length.  And these must be equal to the number of RNGs (=max number of threads).");
    OL_ERROR_PRINTF("No changes to seeds were made.\n");
    return 1;
  }

  NormalRNG::EngineType::result_type seed_value;
  int flag_value;
  for (auto i = 0l, size = num_threads; i < size; ++i) {
    flag_value = boost::python::extract<int>(seed_flag_list[i]);
    if (flag_value) {
      seed_value = boost::python::extract<int>(seed_list[i]);
      normal_rng_vec[i].SetExplicitSeed(seed_value);
    }
  }
  return 0;
}

void RandomnessSourceContainer::PrintState() {
  std::printf("Uniform:\n");
  uniform_generator.PrintState(&std::cout);
  for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
    std::printf("NormalRNG %lu:\n", i);
    normal_rng_vec[i].PrintState(&std::cout);
  }
}

void ExportEnumTypes() {
  boost::python::enum_<OptimizerTypes>("OptimizerTypes", R"%%(
    C++ enums to describe the available optimizers:
    kNull: null optimizer (use for 'dumb' search)
    kGradientDescent: gradient descent
    kNewton: Newton's Method
      )%%")
      .value("null", OptimizerTypes::kNull)
      .value("gradient_descent", OptimizerTypes::kGradientDescent)
      .value("newton", OptimizerTypes::kNewton)
      ;  // NOLINT, this is boost style

  boost::python::enum_<DomainTypes>("DomainTypes", R"%%(
    C++ enums to describe the available domains:
    kTensorProduct: a d-dimensional tensor product domain, D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]
    kSimplex: intersection of kTensorProduct with the unit d-simplex

    The unit d-simplex is defined as the set of x_i such that:
    1) x_i >= 0 \forall i  (i ranging over dimension)
    2) \sum_i x_i <= 1
    (Constrained) optimization is performed over domains.
      )%%")
      .value("tensor_product", DomainTypes::kTensorProduct)
      .value("simplex", DomainTypes::kSimplex)
      ;  // NOLINT, this is boost style

  boost::python::enum_<LogLikelihoodTypes>("LogLikelihoodTypes", R"%%(
    C++ enums to describe the available log likelihood-like measures of model fit:
    kLogMarginalLikelihood: the probability of the observations given [the assumptions of] the model
    kLeaveOneOutLogLikelihood: cross-validation based measure, this indicates how well
      the model explains itself by computing successive log likelihoods, leaving one
      training point out each time.
      )%%")
      .value("log_marginal_likelihood", LogLikelihoodTypes::kLogMarginalLikelihood)
      .value("leave_one_out_log_likelihood", LogLikelihoodTypes::kLeaveOneOutLogLikelihood)
      ;  // NOLINT, this is boost style
}

void ExportOptimizationParameterStructs() {
  boost::python::class_<GradientDescentParameters, boost::noncopyable>("GradientDescentParameters", boost::python::init<int, int, int, double, double, double, double>(R"%%(
    Constructor for a GradientDescentParameters object.

    int num_multistarts: number of initial guesses to try in multistarted gradient descent
    int max_num_steps: maximum number of gradient descent iterations
    int max_num_restarts: maximum number of times we are allowed to call gradient descent.  Should be >= 2 as a minimum.
    double gamma: exponent controlling rate of step size decrease (see get_next_step)
    double pre_mult: scaling factor for step size (see get_next_step)
    double max_relative_change: max relative change allowed per iteration of gradient descent
    double tolerance: when the distance moved btwn steps falls below a factor of this value, stop
    )%%"));

  boost::python::class_<NewtonParameters, boost::noncopyable>("NewtonParameters", boost::python::init<int, int, double, double, double, double>(R"%%(
    Constructor for a NewtonParameters object.

    int num_multistarts: number of initial guesses to try in multistarted newton
    int max_num_steps: maximum number of newton iterations
    double gamma: exponent controlling rate of time_factor growth (see NewtonHyperparameterOptimization)
    double time_factor: initial amount of additive diagonal dominance (see NewtonHyperparameterOptimization())
    double max_relative_change: max relative change allowed per iteration of newton (UNUSED)
    double tolerance: when the magnitude of the gradient falls below this value, stop
    )%%"));
}

void ExportRandomnessContainer() {
  boost::python::class_<RandomnessSourceContainer, boost::noncopyable>("RandomnessSourceContainer", boost::python::init<int>(R"%%(
    Constructor for a RandomnessSourceContainer with enough random sources for at most num_threads simultaneous accesses.

    Random sources are seeded to a repeatable combination of the default seed and the thread id.
    Call SetRandomizedUniformGeneratorSeed() and/or SetRandomizedNormalRNGSeed to use
    an automatically generated (and less repeatable) seed(s).

    INPUTS:
    int num_threads: the max number of threads this object will be used with (sets the number of randomness sources)
    )%%"))
      .def("SetExplicitUniformGeneratorSeed", &RandomnessSourceContainer::SetExplicitUniformGeneratorSeed, R"%%(
    Seeds uniform generator with the specified seed value.

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("SetRandomizedUniformGeneratorSeed", &RandomnessSourceContainer::SetRandomizedUniformGeneratorSeed, R"%%(
    Seeds uniform generator with info dependent on the current time.

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("ResetUniformRNGSeed", &RandomnessSourceContainer::ResetUniformGeneratorState, R"%%(
    Resets Uniform RNG to most recently specified seed value.  Useful for testing
      )%%")
      .def("SetExplicitNormalRNGSeed", &RandomnessSourceContainer::SetExplicitNormalRNGSeed, R"%%(
    Seeds RNG of thread i to f_i(seed, thread_id_i) such that f_i != f_j for i != j.  f_i is repeatable.
    So each thread gets a distinct seed that is easily repeatable for testing.

    NOTE: every thread is GUARANTEED to have a different seed

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("SetRandomizedNormalRNGSeed", &RandomnessSourceContainer::SetRandomizedNormalRNGSeed, R"%%(
    Set a new seed for the random number generator.  A "random" seed is selected based on
    the input seed value, the current time, and the thread_id.

    INPUTS:
    seed: base value for the new seed
    thread_id: id of the thread using this object
      )%%")
      .def("SetNormalRNGSeedPythonList", &RandomnessSourceContainer::SetNormalRNGSeedPythonList, R"%%(
    If seed_flag_list[i] is true, sets the normal rng seed of the i-th thread to the value of seed_list[i].

    If sizes are invalid (i.e., number of seeds != number of generators), then no changes are made and an error code is returned.

    NOTE: Does not guarantee that all threads receive unique seeds!  If that is desired, seed_list should be
          checked BEFORE calling this function.

    RETURNS:
    0 if success, 1 if failure (due to invalid sizes)
      )%%")
      .def("ResetNormalRNGSeed", &RandomnessSourceContainer::ResetNormalRNGState, R"%%(
    Resets all threads' RNGs to their most recently specified seed values.  Useful for testing
      )%%")
      .def("PrintState", &RandomnessSourceContainer::PrintState, R"%%(
    Prints the state of the generator to stdout.  For testing.
      )%%")
      ;  // NOLINT, this is boost style
}

}  // end namespace optimal_learning
