/*!
  \file gpp_python_common.cpp
  \rst
  This file contains definitions of infrequently-used or expensive functions declared in gpp_python_common.hpp.

  Note: several internal functions of this source file are only called from ``Export*()`` functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  ``Export*()`` as Python docstrings, so we saw no need to repeat ourselves.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_common.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <cstdio>  // NOLINT(build/include_order)

#include <vector>  // NOLINT(build/include_order)

#include <boost/python/args.hpp>  // NOLINT(build/include_order)
#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/enum.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_optimizer_parameters.hpp"
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

RandomnessSourceContainer::RandomnessSourceContainer(int num_threads)
    : uniform_generator(kUniformDefaultSeed),
      normal_rng_vec(num_threads),
      num_normal_rng_(num_threads) {
  SetExplicitNormalRNGSeed(kNormalDefaultSeed);
}

void RandomnessSourceContainer::SetExplicitUniformGeneratorSeed(NormalRNG::EngineType::result_type seed) {
  uniform_generator.SetExplicitSeed(seed);
}

void RandomnessSourceContainer::SetRandomizedUniformGeneratorSeed(NormalRNG::EngineType::result_type seed) {
  uniform_generator.SetRandomizedSeed(seed, 0);  // single instance, so thread_id = 0
}

void RandomnessSourceContainer::ResetUniformGeneratorState() {
  uniform_generator.ResetToMostRecentSeed();
}

void RandomnessSourceContainer::SetExplicitNormalRNGSeed(NormalRNG::EngineType::result_type seed) {
  for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
    normal_rng_vec[i].SetExplicitSeed(seed + i);
  }
}

void RandomnessSourceContainer::SetRandomizedNormalRNGSeed(NormalRNG::EngineType::result_type seed) {
  for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
    normal_rng_vec[i].SetRandomizedSeed(seed, i);
  }
}

bool RandomnessSourceContainer::SetNormalRNGSeedPythonList(const boost::python::list& seed_list, const boost::python::list& seed_flag_list) {
  auto seed_list_len = boost::python::len(seed_list);
  auto seed_flag_list_len = boost::python::len(seed_flag_list);
  IdentifyType<decltype(seed_flag_list_len)>::type num_threads = normal_rng_vec.size();
  if (unlikely(seed_list_len != seed_flag_list_len || seed_list_len != num_threads)) {
    OL_ERROR_PRINTF("NORMAL RNG SEEDING ERROR: len(seed_list) = %lu, len(seed_flag_list) = %lu, normal_rng_vec.size() = %lu\n", seed_list_len, seed_flag_list_len, num_threads);
    OL_ERROR_PRINTF("Seed list & flag list must be the same length.  And these must be equal to the number of RNGs (=max number of threads).");
    OL_ERROR_PRINTF("No changes to seeds were made.\n");
    return false;
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
  return true;
}

void RandomnessSourceContainer::ResetNormalRNGState() {
  for (auto& entry : normal_rng_vec) {
    entry.ResetToMostRecentSeed();
  }
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

    * ``kNull``: null optimizer (use for 'dumb' search)
    * ``kGradientDescent``: gradient descent
    * ``kNewton``: Newton's Method
      )%%")
      .value("null", OptimizerTypes::kNull)
      .value("gradient_descent", OptimizerTypes::kGradientDescent)
      .value("newton", OptimizerTypes::kNewton)
      ;  // NOLINT, this is boost style

  boost::python::enum_<DomainTypes>("DomainTypes", R"%%(
    C++ enums to describe the available domains:

    * ``kTensorProduct``: a d-dimensional tensor product domain, ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``
    * ``kSimplex``: intersection of kTensorProduct with the unit d-simplex

    The unit d-simplex is defined as the set of ``x_i`` such that:

    1. ``x_i >= 0 \forall i``  (i ranging over dimension)
    2. ``\sum_i x_i <= 1``

    (Constrained) optimization is performed over domains.
      )%%")
      .value("tensor_product", DomainTypes::kTensorProduct)
      .value("simplex", DomainTypes::kSimplex)
      ;  // NOLINT, this is boost style

  boost::python::enum_<LogLikelihoodTypes>("LogLikelihoodTypes", R"%%(
    C++ enums to describe the available log likelihood-like measures of model fit:

    * ``kLogMarginalLikelihood``: the probability of the observations given [the assumptions of] the model
    * ``kLeaveOneOutLogLikelihood``: cross-validation based measure, this indicates how well
      the model explains itself by computing successive log likelihoods, leaving one
      training point out each time.
      )%%")
      .value("log_marginal_likelihood", LogLikelihoodTypes::kLogMarginalLikelihood)
      .value("leave_one_out_log_likelihood", LogLikelihoodTypes::kLeaveOneOutLogLikelihood)
      ;  // NOLINT, this is boost style
}

void ExportOptimizerParameterStructs() {
  boost::python::class_<GradientDescentParameters, boost::noncopyable>("GradientDescentParameters", boost::python::init<int, int, int, int, double, double, double, double>(
      (boost::python::arg("num_multistarts"), "max_num_steps", "max_num_restarts", "num_steps_averaged", "gamma", "pre_mult", "max_relative_change", "tolerance"), R"%%(
    Constructor for a GradientDescentParameters object.

    :param num_multistarts: number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
    :type num_multistarts: int > 0
    :param max_num_steps: maximum number of gradient descent iterations per restart (suggest: 200-1000)
    :type max_num_steps: int > 0
    :param max_num_restarts: maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 4-20)
    :type max_num_restarts: int > 0
    :param num_steps_averaged: number of steps to use in polyak-ruppert averaging (see above) (suggest: 10-50% of max_num_steps for stochastic problems, 0 otherwise) (UNUSED)
    :type num_steps_averaged: int (range is clamped as described above)
    :param gamma: exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
    :type gamma: float64 > 1.0
    :param pre_mult: scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
    :type pre_mult: float64 > 0.0
    :param max_relative_change: max change allowed per GD iteration (as a relative fraction of current distance to wall)
        (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
    :type max_relative_change: float64 in [0, 1]
    :param tolerance: when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
        (e.g., at a boundary), stop.  (suggest: 1.0e-7)
    :type tolerance: float64 >= 0.0
    )%%"))
      .def_readwrite("num_multistarts", &GradientDescentParameters::num_multistarts, "number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)")
      .def_readwrite("max_num_steps", &GradientDescentParameters::max_num_steps, "maximum number of gradient descent iterations per restart (suggest: 200-1000)")
      .def_readwrite("max_num_restarts", &GradientDescentParameters::max_num_restarts, "maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 4-20)")
      .def_readwrite("num_steps_averaged", &GradientDescentParameters::num_steps_averaged, "number of steps to use in polyak-ruppert averaging (suggest: 10-50% of max_num_steps for stochastic problems, 0 otherwise)")
      .def_readwrite("gamma", &GradientDescentParameters::gamma, "exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)")
      .def_readwrite("pre_mult", &GradientDescentParameters::pre_mult, "scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)")
      .def_readwrite("max_relative_change", &GradientDescentParameters::max_relative_change, "max change allowed per GD iteration (as a relative fraction of current distance to wall), see ctor docstring")
      .def_readwrite("tolerance", &GradientDescentParameters::tolerance, "when the magnitude of the gradient falls below this value OR we will not move farther than tolerance")
      ;  // NOLINT, this is boost style

  boost::python::class_<NewtonParameters, boost::noncopyable>("NewtonParameters", boost::python::init<int, int, double, double, double, double>(
      (boost::python::arg("num_multistarts"), "max_num_steps", "gamma", "time_factor", "max_relative_change", "tolerance"), R"%%(
    Constructor for a NewtonParameters object.

    :param num_multistarts: number of initial guesses to try in multistarted newton (suggest: a few hundred)
    :type num_multistarts: int > 0
    :param max_num_steps: maximum number of newton iterations (per initial guess) (suggest: 100)
    :type max_num_steps: int > 0
    :param gamma: exponent controlling rate of time_factor growth (see function comments) (suggest: 1.01-1.1)
    :type gamma: float64 > 1.0
    :param time_factor: initial amount of additive diagonal dominance (see function comments) (suggest: 1.0e-3-1.0e-1)
    :type time_factor: float64 > 0.0
    :param max_relative_change: max change allowed per update (as a relative fraction of current distance to wall) (suggest: 1.0)
    :type max_relative_change: float64 in [0, 1]
    :param tolerance: when the magnitude of the gradient falls below this value, stop (suggest: 1.0e-10)
    :type tolerance: float64 >= 0.0
    )%%"))
      .def_readwrite("num_multistarts", &NewtonParameters::num_multistarts, "number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)")
      .def_readwrite("max_num_steps", &NewtonParameters::max_num_steps, "maximum number of gradient descent iterations per restart (suggest: 200-1000)")
      .def_readwrite("gamma", &NewtonParameters::gamma, "exponent controlling rate of time_factor growth (see class docs and NewtonOptimizer) (suggest: 1.01-1.1)")
      .def_readwrite("time_factor", &NewtonParameters::time_factor, "initial amount of additive diagonal dominance (see class docs and NewtonOptimizer) (suggest: 1.0e-3-1.0e-1)")
      .def_readwrite("max_relative_change", &NewtonParameters::max_relative_change, "max change allowed per update (as a relative fraction of current distance to wall) (Newton may ignore this) (suggest: 1.0)")
      .def_readwrite("tolerance", &NewtonParameters::tolerance, "when the magnitude of the gradient falls below this value, stop (suggest: 1.0e-10)")
      ;  // NOLINT, this is boost style
}

void ExportRandomnessContainer() {
  boost::python::class_<RandomnessSourceContainer, boost::noncopyable>("RandomnessSourceContainer", boost::python::init<int>(R"%%(
    Constructor for a RandomnessSourceContainer with enough random sources for at most num_threads simultaneous accesses.

    Random sources are seeded to a repeatable combination of the default seed and the thread id.
    Call SetRandomizedUniformGeneratorSeed() and/or SetRandomizedNormalRNGSeed to use
    an automatically generated (and less repeatable) seed(s).

    :param num_threads: the max number of threads this object will be used with (sets the number of randomness sources)
    :type num_threads: int
    )%%"))
      .add_property("num_normal_rng", &RandomnessSourceContainer::num_normal_rng, R"%%(
    Return the number of NormalRNG objects being tracked.

    This is the maximum number of EI evaluations (threads) this object can support.
      )%%")
      .def("SetExplicitUniformGeneratorSeed", &RandomnessSourceContainer::SetExplicitUniformGeneratorSeed, R"%%(
    Seeds uniform generator with the specified seed value.

    :param seed: base seed value to use
    :type seed: unsigned int
      )%%")
      .def("SetRandomizedUniformGeneratorSeed", &RandomnessSourceContainer::SetRandomizedUniformGeneratorSeed, R"%%(
    Seeds uniform generator with info dependent on the current time.

    :param seed: base seed value to use
    :type seed: unsigned int
      )%%")
      .def("ResetUniformRNGSeed", &RandomnessSourceContainer::ResetUniformGeneratorState, R"%%(
    Resets Uniform RNG to most recently specified seed value.  Useful for testing.
      )%%")
      .def("SetExplicitNormalRNGSeed", &RandomnessSourceContainer::SetExplicitNormalRNGSeed, R"%%(
    Seeds RNG of thread ``i`` to ``f_i(seed, thread_id_i)`` such that ``f_i != f_j`` for ``i != j``.  ``f_i`` is repeatable.
    So each thread gets a distinct seed that is easily repeatable for testing.

    .. NOTE:: every thread is GUARANTEED to have a different seed

    :param seed: base seed value to use
    :type seed: unsigned int
      )%%")
      .def("SetRandomizedNormalRNGSeed", &RandomnessSourceContainer::SetRandomizedNormalRNGSeed, R"%%(
    Set a new seed for the random number generator.  A "random" seed is selected based on
    the input seed value, the current time, and the thread_id.

    :param seed: base seed value to use
    :type seed: unsigned int
    :param thread_id: id of the thread using this object
    :type thread_id: int >= 0
      )%%")
      .def("SetNormalRNGSeedPythonList", &RandomnessSourceContainer::SetNormalRNGSeedPythonList, R"%%(
    If ``seed_flag_list[i]`` is true, sets the normal rng seed of the ``i``-th thread to the value of ``seed_list[i]``.

    If sizes are invalid (i.e., number of seeds != number of generators), then no changes are made and an error code is returned.

    .. NOTE:: Does not guarantee that all threads receive unique seeds!  If that is desired, seed_list should be
        checked BEFORE calling this function.

    :return: true if success, false if failure (due to invalid sizes)
    :rtype: bool
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
