/*!
  \file gpp_python_common.hpp
  \rst
  Classes and utilities that are useful throughout the gpp_python_* interface code.

  This file includes three main groups of things:

  1. Tools for translating between C++ and Python data sources

     a. PythonInterfaceInputContainer: captures the most common set of inputs used in gpp_python
     b. utilities for copying between std::vector and boost::python::list

  2. A RandomnessSourceContainer for moving consistent RNG state between C++, Python
  3. Export*() functions for giving Python access to various C++ calls via boost::python.

     a. enum classes
     b. optimizer parameter structs
     c. RandomnessSourceContainer from item 2)

  Functions callable from Python generally have the following form:

  1. Copy vector inputs from references of Python structures to C++ (e.g., boost::python::list to std::vector).
     Items 1a) and 1b) are helpful for this.
  2. Construct any temporary objects needed by C++ (e.g., Evaluator/State pairs).
  3. Compute the desired result with C++ calls.
  4. Copy/return the desired result from C++ container back into a boost::python::list (or return directly for primitive types)
     Some of the functions in item 1b) are useful for this.

  See other gpp_python_*.cpp files for examples (e.g., gpp_python_expected_improvement.cpp or gpp_python_model_selection.cpp).

  General notes about the Python interface:

  1. We use raw strings (C++11) to pass multiline string
     literals to boost to specify python docstrings. Our delimiter is: ``%%``. The
     format is: ``R"%%(put anything here, no need to escape chars)%%"``.
     Unfortunately this confuses linters but it is much more human-readable than alternatives.

  2. ALL ARRAYS/LISTS MUST BE FLATTENED!
     What that means:
     Matrices will be described as A[dim1][dim2]...[dimN]
     To FLATTEN a matrix is to lay it out in memory C-style;
     i.e., rightmost index varies the most rapidly.
     For example: ``A[3][4] =``
     ``[4  32  5  2``
     `` 53 12  8  1``
     `` 81  2  93 0]``
     would be FLATTENED into an array:
     ``A_flat[12] = [4 32 5 2 53 12 8 1 81 2 93 0]``
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_COMMON_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_COMMON_HPP_

#include <vector>

#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

/*!\rst
  Container class for translating a standard set of python (list) inputs into std::vector.
\endrst*/
struct PythonInterfaceInputContainer {
  /*!\rst
    Minimal constructor that sets up ``points_to_sample`` only; generally used when a
    GaussianProcess object is already available (e.g., in GaussianProcess member function wrappers).

    \param
      :points_to_sample[num_to_sample][dim]: points at which to compute GP quantities
      :dim: number of spatial dimension (independent parameters)
      :num_to_sample: number of points being sampled from the GP
  \endrst*/
  PythonInterfaceInputContainer(const boost::python::list& points_to_sample_in, int dim_in, int num_to_sample_in);

  /*!\rst
    Minimal constructor that sets up ``points_to_sample`` and ``points_being_sampled``; generally used when a
    GaussianProcess object is already available (e.g., in Expected Improvement wrappers).

    \param
      :points_to_sample[num_to_sample][dim]: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :pylist points_being_sampled_in[num_being_sampled][dim]: points that are being sampled in concurrent experiments
      :dim: number of spatial dimension (independent parameters)
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
      :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  \endrst*/
  PythonInterfaceInputContainer(const boost::python::list& points_to_sample_in, const boost::python::list& points_being_sampled_in, int dim_in, int num_to_sample_in, int num_being_sampled_in);

  /*!\rst
    Constructor that sets up GP-related members held in this container; generally used when a GaussianProcess object is not
    available/relevant (or to construct a GP). Does not set points_being_sampled.

    \param
      :pylist hyperparameters_in[2]: [0]: \alpha: the hyperparameter \alpha, (e.g., signal variance, \sigma_f^2)
                                     [1]: lengths[dim]: the hyperparameter length scales, one per spatial dimension
      :pylist points_sampled_in[num_sampled][dim]: points that have already been sampled
      :pylist points_sampled_value_in[num_sampled]: objective values of the already-sampled points
      :pylist noise_variance_in[num_sampled]: the \sigma_n^2 (noise variance) associated w/observation, points_sampled_value
      :pylist points_to_sample_in[num_to_sample][dim]: points at which to compute GP quantities
      :dim: number of spatial dimension (independent parameters)
      :num_sampled: number of already-sampled points
      :num_to_sample: number of points being sampled from the GP
  \endrst*/
  PythonInterfaceInputContainer(const boost::python::list& hyperparameters_in, const boost::python::list& points_sampled_in, const boost::python::list& points_sampled_value_in, const boost::python::list& noise_variance_in, const boost::python::list& points_to_sample_in, int dim_in, int num_sampled_in, int num_to_sample_in);

  int dim;
  int num_sampled;
  int num_to_sample;
  int num_being_sampled;
  double alpha;
  std::vector<double> lengths;
  std::vector<double> points_sampled;
  std::vector<double> points_sampled_value;
  std::vector<double> noise_variance;
  std::vector<double> points_to_sample;
  std::vector<double> points_being_sampled;
};

/*!\rst
  Container for randomness sources to be used with the python interface.  Python should create a singleton of this object
  and then pass it back to any C++ function requiring randomness sources.  Outside of testing, we only want a single version
  of this object floating around in order for all RNGs to remain consistent.

  Python generally has minimal interaction with this class; it is meant to be constructed in Python and then passed back to
  C++ for use.

  This class will track enough enough sources so that multithreaded computation is well-defined.

  This class exposes its member functions directly to Python; these member functions are for setting and resetting seed
  values for the randomness sources.
\endrst*/
class RandomnessSourceContainer {
  static constexpr NormalRNG::EngineType::result_type kNormalDefaultSeed = 314;
  static constexpr UniformRandomGenerator::EngineType::result_type kUniformDefaultSeed = 314;

 public:
  /*!\rst
    Creates the randomness container with enough random sources for at most num_threads simultaneous accesses.

    Random sources are seeded to a repeatable combination of the default seed and the thread id.
  \endrst*/
  explicit RandomnessSourceContainer(int num_threads);

  /*!\rst
    Get the current number of threads being tracked.  Can be less than normal_rng_vec.size()
  \endrst*/
  int num_normal_rng() {
    return num_normal_rng_;
  }

  /*!\rst
    Seeds uniform generator with the specified seed value

    \param
      :seed: base seed value to use
  \endrst*/
  void SetExplicitUniformGeneratorSeed(NormalRNG::EngineType::result_type seed);

  /*!\rst
    Seeds uniform generator with current time information

    \param
      :seed: base seed value to use
  \endrst*/
  void SetRandomizedUniformGeneratorSeed(NormalRNG::EngineType::result_type seed);

  /*!\rst
    Resets uniform generator to its most recently used seed.
  \endrst*/
  void ResetUniformGeneratorState();

  /*!\rst
    Seeds RNG of thread i to f_i(seed, thread_id_i) such that f_i != f_j for i != j.  f_i is repeatable.
    so each thread gets a distinct seed that is easily repeatable for testing

    .. NOTE:: every thread is GUARANTEED to have a different seed

    \param
      :seed: base seed value to use
  \endrst*/
  void SetExplicitNormalRNGSeed(NormalRNG::EngineType::result_type seed);

  /*!\rst
    Seeds each thread with a combination of current time, thread_id, and (potentially) other factors.
    multiple calls to this should produce different seeds modulo aliasing issues

    \param
      :seed: base seed value to use
  \endrst*/
  void SetRandomizedNormalRNGSeed(NormalRNG::EngineType::result_type seed);

  /*!\rst
    If ``seed_flag_list[i]`` is true, sets the normal rng seed of the ``i``-th thread to the value of ``seed_list[i]``.

    If sizes are invalid (i.e., number of seeds != number of generators), then no changes are made and an error code is returned.

    .. NOTE:: Does not guarantee that all threads receive unique seeds!  If that is desired, seed_list should be
              checked BEFORE calling this function.

    \return
      true if successful, false otherwise (due to invalid sizes)
  \endrst*/
  bool SetNormalRNGSeedPythonList(const boost::python::list& seed_list, const boost::python::list& seed_flag_list);

  /*!\rst
    Resets all threads' RNGs to the seed values they were initialized with.  Useful for testing.
  \endrst*/
  void ResetNormalRNGState();

  void PrintState();

  //! The uniform random generator that will be used by C++ to make uniform draws.
  UniformRandomGenerator uniform_generator;
  //! The normal random generators (one per thread) that will be used by C++ to make N(0, 1)-distributed draws.
  std::vector<NormalRNG> normal_rng_vec;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(RandomnessSourceContainer);

 private:
  //! length of normal_rng_vec
  int num_normal_rng_;
};

/*!\rst
  Copies the first doubles elements of a python list (input) into a std::vector (output)
  Resizes output if needed.

  .. WARNING:: undefined behavior if the python list contains anything except type ``double``!

  \param
    :input: python list to copy from
    :size: number of elements to copy
  \output
    :output: std::vector with copies of the first size items of input
\endrst*/
void CopyPylistToVector(const boost::python::list& input, int size, std::vector<double>& output);

/*!\rst
  Copies the first size [min, max] pairs from input to output.
  Size of input MUST be 2*size.

  .. WARNING:: undefined behavior if the python list contains anything except type double!

  \param
    :input: python list to copy from
    :size: number of pairs to copy
  \output
    :output: std::vector with copies of the first size items of input
\endrst*/
void CopyPylistToClosedIntervalVector(const boost::python::list& input, int size, std::vector<ClosedInterval>& output);

/*!\rst
  Produces a PyList with the same size as the input vector and that is
  element-wise equal to the input vector.

  \param
    :input: std::vector to be copied
  \return
    python list that is element-wise equivalent to input
\endrst*/
boost::python::list VectorToPylist(const std::vector<double>& input);

/*!\rst
  Export C++'s enum classes to Python; e.g., DomainTypes, OptimizerTypes, etc. Includes docstrings.
\endrst*/
void ExportEnumTypes();

/*!\rst
  Export the parameter structs from gpp_optimizer_parameters.hpp to Python. Includes docstrings.
\endrst*/
void ExportOptimizerParameterStructs();

/*!\rst
  Export the class RandomnessSourceContainer and its member functions to Python. Includes docstrings.
\endrst*/
void ExportRandomnessContainer();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_COMMON_HPP_
