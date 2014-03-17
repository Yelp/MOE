// gpp_python.cpp
/*
  This file contains code used to provide a interface to Python.  In particular, a number of lower level GP functions
  (e.g., mean, variance, gradients thereof) are exposed for testing purposes.  This also exposes functions to compute
  and optimize EI as well as compute/optimize log likelihood validators.

  The form of the python interface is specified and documented at the bottom of this file, in BOOST_PYTHON_MODULE(GPP).
  For the most part, the function call interfaces are self explanatory.  To promote data re-use and to deal with issues of
  persistence (for RNGs), this file also defines a few container classes and exposes classes from C++.  Python generally has
  minimal interaction with these classes; they're meant to be constructed in Python and then passed back to C++ for use.

  Functions callable from Python generally have the following form:
  1) Copy vector inputs from boost::python::list references into C++ std::vector.
  2) Construct any temporary objects needed by C++ (e.g., state containers)
  3) Compute the desired result
  4) Copy/return the desired result from C++ container back into a boost::python::list (or return directly for primitive types)
  There are a few (messy) utilities at the top of the file that capture the most common elements of steps 1) and 4) to reduce
  boilerplate.
*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <cstdio>  // NOLINT(build/include_order)

#include <algorithm>  // NOLINT(build/include_order)
#include <limits>  // NOLINT(build/include_order)
#include <stdexcept>  // NOLINT(build/include_order)
#include <string>  // NOLINT(build/include_order)
#include <type_traits>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include <boost/python/handle.hpp>  // NOLINT(build/include_order)
#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/def.hpp>  // NOLINT(build/include_order)
#include <boost/python/dict.hpp>  // NOLINT(build/include_order)
#include <boost/python/docstring_options.hpp>  // NOLINT(build/include_order)
#include <boost/python/enum.hpp>  // NOLINT(build/include_order)
#include <boost/python/errors.hpp>  // NOLINT(build/include_order)
#include <boost/python/exception_translator.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)
#include <boost/python/make_constructor.hpp>  // NOLINT(build/include_order)
#include <boost/python/module.hpp>  // NOLINT(build/include_order)
#include <boost/python/object.hpp>  // NOLINT(build/include_order)
#include <boost/python/scope.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_covariance_test.hpp"
#include "gpp_domain.hpp"
#include "gpp_domain_test.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_geometry_test.hpp"
#include "gpp_linear_algebra_test.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_math_test.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization_test.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_optimization_test.hpp"
#include "gpp_random.hpp"
#include "gpp_random_test.hpp"

namespace {

/*
  Copies the first doubles elements of a python list (input) into a std::vector (output)
  Resizes output if needed.

  WARNING: undefined behavior if the python list contains anything except type double!

  INPUTS:
  input: python list to copy from
  size: number of elements to copy
  OUTPUTS:
  output: std::vector with copies of the first size items of input
*/
void CopyPylistToVector(const boost::python::list& input, int size, std::vector<double>& output) {
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    output[i] = boost::python::extract<double>(input[i]);
  }
}

/*
  Copies the first size [min, max] pairs from input to output.
  Size of input MUST be 2*size.

  WARNING: undefined behavior if the python list contains anything except type double!

  INPUTS:
  input: python list to copy from
  size: number of pairs to copy
  OUTPUTS:
  output: std::vector with copies of the first size items of input
*/
void CopyPylistToClosedIntervalVector(const boost::python::list& input, int size, std::vector<optimal_learning::ClosedInterval>& output) {
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    output[i].min = boost::python::extract<double>(input[2*i + 0]);
    output[i].max = boost::python::extract<double>(input[2*i + 1]);
  }
}

/*
  Produces a PyList with the same size as the input vector and that is
  element-wise equal to the input vector.

  INPUTS:
  input: std::vector to be copied
  RETURNS:
  python list that is element-wise equivalent to input
*/
boost::python::list VectorToPylist(const std::vector<double>& input) {
  boost::python::list result;
  for (const auto& entry : input) {
    result.append(entry);
  }
  return result;
}

}  // end unnamed namespace

namespace optimal_learning {

/*
  Container class for translating a standard set of python (list) inputs into std::vector.
*/
class PythonInterfaceInputContainer {
 public:
  /*
     Minimal constructor that only sets up points_to_sample; generally used when a GaussianProcess object is already available.

     INPUTS:
     pylist points_to_sample_in[num_to_sample][dim]: points to sample
     dim: number of spatial dimension (independent parameters)
     num_to_sample: number of points to sample
  */
  PythonInterfaceInputContainer(const boost::python::list& points_to_sample_in, int dim_in, int num_to_sample_in) :
      dim(dim_in),
      num_sampled(0),
      num_to_sample(num_to_sample_in),
      alpha(0.0),
      lengths(0),
      points_sampled(0),
      points_sampled_value(0),
      noise_variance(0),
      points_to_sample(dim*num_to_sample) {
    CopyPylistToVector(points_to_sample_in, dim*num_to_sample, points_to_sample);
  }

  /*
     Full constructor that sets up all members held in this container; generally used when a GaussianProcess object is not
     available/relevant (or to construct a GP).

     INPUTS:
     pylist points_to_sample_in[num_to_sample][dim]: points to sample
     dim: number of spatial dimension (independent parameters)
     num_to_sample: number of points to sample
  */
  PythonInterfaceInputContainer(const boost::python::list& hyperparameters_in, const boost::python::list& points_sampled_in, const boost::python::list& points_sampled_value_in, const boost::python::list& noise_variance_in, const boost::python::list& points_to_sample_in, int dim_in, int num_sampled_in, int num_to_sample_in) :
      dim(dim_in),
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

  int dim;
  int num_sampled;
  int num_to_sample;
  double alpha;
  std::vector<double> lengths;
  std::vector<double> points_sampled;
  std::vector<double> points_sampled_value;
  std::vector<double> noise_variance;
  std::vector<double> points_to_sample;
};

/*
  Container for randomness sources to be used with the python interface.  Python should create a singleton of this object
  and then pass it back to any C++ function requiring randomness sources.

  This class will track enough enough sources so that multithreaded computation is well-defined.

  This class exposes its member functions directly to Python; these member functions are for setting and resetting seed
  values for the randomness sources.
*/
class RandomnessSourceContainer {
  static constexpr NormalRNG::EngineType::result_type kNormalDefaultSeed = 314;
  static constexpr UniformRandomGenerator::EngineType::result_type kUniformDefaultSeed = 314;

 public:
  /*
    Creates the randomness container with enough random sources for at most num_threads simultaneous accesses.

    Random sources are seeded to a repeatable combination of the default seed and the thread id.
  */
  explicit RandomnessSourceContainer(int num_threads) : uniform_generator(kUniformDefaultSeed), normal_rng_vec(num_threads), num_normal_rng_(num_threads) {
    SetExplicitNormalRNGSeed(kNormalDefaultSeed);
  }

  /*
    Get the current number of threads being tracked.  Can be less than normal_rng_vec.size()
  */
  int num_normal_rng() {
    return num_normal_rng_;
  }

  /*
    Seeds uniform generator with the specified seed value

    INPUTS:
    seed: base seed value to use
  */
  void SetExplicitUniformGeneratorSeed(NormalRNG::EngineType::result_type seed) {
    uniform_generator.SetExplicitSeed(seed);
  }

  /*
    Seeds uniform generator with current time information

    INPUTS:
    seed: base seed value to use
  */
  void SetRandomizedUniformGeneratorSeed(NormalRNG::EngineType::result_type seed) {
    uniform_generator.SetRandomizedSeed(seed, 0);  // single instance, so thread_id = 0
  }

  /*
    Resets uniform generator to its most recently used seed.
  */
  void ResetUniformGeneratorState() {
    uniform_generator.ResetToMostRecentSeed();
  }

  /*
    seeds RNG of thread i to f_i(seed, thread_id_i) such that f_i != f_j for i != j.  f_i is repeatable.
    so each thread gets a distinct seed that is easily repeatable for testing

    NOTE: every thread is GUARANTEED to have a different seed

    INPUTS:
    seed: base seed value to use
  */
  void SetExplicitNormalRNGSeed(NormalRNG::EngineType::result_type seed) {
    for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
      normal_rng_vec[i].SetExplicitSeed(seed + i);
    }
  }

  /*
    seeds each thread with a combination of current time, thread_id, and (potentially) other factors.
    multiple calls to this should produce different seeds modulo aliasing issues

    INPUTS:
    seed: base seed value to use
  */
  void SetRandomizedNormalRNGSeed(NormalRNG::EngineType::result_type seed) {
    for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
      normal_rng_vec[i].SetRandomizedSeed(seed, i);
    }
  }

  /*
    If seed_flag_list[i] is true, sets the normal rng seed of the i-th thread to the value of seed_list[i].

    If sizes are invalid (i.e., number of seeds != number of generators), then no changes are made and an error code is returned.

    NOTE: Does not guarantee that all threads receive unique seeds!  If that is desired, seed_list should be
          checked BEFORE calling this function.

    RETURNS:
    0 if success, 1 if failure (due to invalid sizes)
  */
  int SetNormalRNGSeedPythonList(const boost::python::list& seed_list, const boost::python::list& seed_flag_list) {
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

  /*
    Resets all threads' RNGs to the seed values they were initialized with.  Useful for testing
  */
  void ResetNormalRNGState() {
    for (auto& entry : normal_rng_vec) {
      entry.ResetToMostRecentSeed();
    }
  }

  void PrintState() {
    std::printf("Uniform:\n");
    uniform_generator.PrintState(&std::cout);
    for (IdentifyType<decltype(normal_rng_vec)>::type::size_type i = 0, size = normal_rng_vec.size(); i < size; ++i) {
      std::printf("NormalRNG %lu:\n", i);
      normal_rng_vec[i].PrintState(&std::cout);
    }
  }

  UniformRandomGenerator uniform_generator;
  std::vector<NormalRNG> normal_rng_vec;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(RandomnessSourceContainer);

 private:
  int num_normal_rng_;
};

boost::python::list get_mean_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_mean(input_container.num_to_sample);
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(), input_container.num_to_sample, false);
  gaussian_process.ComputeMeanOfPoints(points_to_sample_state, to_sample_mean.data());

  return VectorToPylist(to_sample_mean);
}

boost::python::list get_grad_mean_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_grad_mean(input_container.dim*input_container.num_to_sample);
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(), input_container.num_to_sample, true);
  gaussian_process.ComputeGradMeanOfPoints(points_to_sample_state, to_sample_grad_mean.data());

  return VectorToPylist(to_sample_grad_mean);
}

boost::python::list get_var_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_var(Square(input_container.num_to_sample));
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(), input_container.num_to_sample, false);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, to_sample_var.data());

  boost::python::list result;

  // copy lower triangle of chol_var into its upper triangle b/c python expects a proper symmetric matrix
  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < i; ++j) {
      to_sample_var[i*num_to_sample + j] = to_sample_var[j*num_to_sample + i];
    }
  }

  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < num_to_sample; ++j) {
      result.append(to_sample_var[j*num_to_sample + i]);
    }
  }

  return result;
}

boost::python::list get_chol_var_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> chol_var(Square(input_container.num_to_sample));
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(), input_container.num_to_sample, false);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, chol_var.data());
  ComputeCholeskyFactorL(num_to_sample, chol_var.data());

  boost::python::list result;

  // zero upper triangle of chol_var b/c python expects a proper triangular matrix
  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < i; ++j) {
      chol_var[i*num_to_sample + j] = 0.0;
    }
  }

  for (int i = 0; i < num_to_sample; ++i) {
    for (int j = 0; j < num_to_sample; ++j) {
      result.append(chol_var[j*num_to_sample + i]);
    }
  }

  return result;
}

boost::python::list get_grad_var_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample, int var_of_grad) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  std::vector<double> to_sample_grad_var(input_container.dim*Square(input_container.num_to_sample));
  std::vector<double> chol_var(Square(input_container.num_to_sample));
  GaussianProcess::StateType points_to_sample_state(gaussian_process, input_container.points_to_sample.data(), input_container.num_to_sample, true);
  gaussian_process.ComputeVarianceOfPoints(&points_to_sample_state, chol_var.data());
  ComputeCholeskyFactorL(input_container.num_to_sample, chol_var.data());
  gaussian_process.ComputeGradCholeskyVarianceOfPoints(&points_to_sample_state, var_of_grad, chol_var.data(), to_sample_grad_var.data());

  return VectorToPylist(to_sample_grad_var);
}

double compute_expected_improvement_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample, int num_its, double best_so_far, bool force_monte_carlo, RandomnessSourceContainer& randomness_source) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  if ((num_to_sample == 1) && (force_monte_carlo == false)) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, input_container.points_to_sample.data(), input_container.num_to_sample, false, nullptr);
    return ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, num_its, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, input_container.points_to_sample.data(), input_container.num_to_sample, false, randomness_source.normal_rng_vec.data());
    return ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }
}

boost::python::list compute_grad_expected_improvement_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample, int num_its, double best_so_far, bool force_monte_carlo, RandomnessSourceContainer& randomness_source, const boost::python::list& current_point) {
  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);

  // this vector stores all points_to_sample in a contiguous space
  // the first point is "current_point", which is the point at which we are differentiating
  // the remaining point(s) are from "points_to_sample", which are the other simultaneously running experiments we need to
  // account for
  std::vector<double> union_of_points((input_container.num_to_sample + 1) * input_container.dim);
  CopyPylistToVector(current_point, input_container.dim, union_of_points);

  std::vector<double> grad_EI(input_container.dim);
  if ((num_to_sample == 0) && (force_monte_carlo == false)) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
    // here num_to_sample = 0, so union_of_points contains just current_point
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), input_container.num_to_sample + 1, true, nullptr);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_EI.data());
  } else {
    // copy over points_to_sample as described above (at the declaration of union_of_points)
    std::copy(input_container.points_to_sample.begin(), input_container.points_to_sample.end(), union_of_points.begin() + input_container.dim);

    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, num_its, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), input_container.num_to_sample + 1, true, randomness_source.normal_rng_vec.data());
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_EI.data());
  }

  return VectorToPylist(grad_EI);
}

namespace {

/*
  Utility that dispatches EI optimization based on optimizer type and num_samples_to_generate.
  This is just used to reduce copy-pasted code.

  INPUTS:
  optimization_parameters: round_generation/MOE_driver.MOERunner.ExpectedImprovementOptimizationParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
  gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
    underlying GP
  input_container: PythonInterfaceInputContainer object containing data about points_to_sample
  domain: object specifying the domain to optimize over (see gpp_domain.hpp)
  domain_name: name of the domain, e.g., "tensor" or "simplex". Used to update the status dict
  optimizer_type: type of optimization to use (e.g., null, gradient descent)
  num_samples_to_generate: how many simultaneous experiments you would like to run
  best_so_far: value of the best sample so far (must be min(points_sampled_value))
  max_int_steps: maximum number of MC iterations
  max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
  randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
  status: pydict object; cannot be None

  OUTPUTS:
  randomness_source: PRNG internal states modified
  status: modified on exit to describe whether convergence occurred
  best_points_to_sample[num_samples_to_generate][dim]: next set of points to evaluate
*/
template <typename DomainType>
void DispatchExpectedImprovementOptimization(const boost::python::object& optimization_parameters, const GaussianProcess& gaussian_process, const PythonInterfaceInputContainer& input_container, const DomainType& domain_object, const std::string& domain_name, OptimizerTypes optimizer_type, int num_samples_to_generate, double best_so_far, int max_int_steps, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status, double * restrict best_points_to_sample) {
  bool found_flag = false;
  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      // optimization_parameters must contain an int num_multistarts field, extract it
      int num_random_samples = boost::python::extract<int>(optimization_parameters.attr("num_random_samples"));

      if (num_samples_to_generate == 1) {
        ComputeOptimalPointToSampleViaLatinHypercubeSearch(gaussian_process, domain_object, input_container.points_to_sample.data(), num_random_samples, input_container.num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, &randomness_source.uniform_generator, randomness_source.normal_rng_vec.data(), best_points_to_sample);
      } else {
        bool random_search_only = true;
        GradientDescentParameters gradient_descent_parameters(0, 0, 0, 1.0, 1.0, 1.0, 0.0);  // dummy struct; we aren't using gradient descent
        ComputeOptimalSetOfPointsToSample(gaussian_process, gradient_descent_parameters, domain_object, input_container.points_to_sample.data(), input_container.num_to_sample, best_so_far, max_int_steps, max_num_threads, random_search_only, num_random_samples, num_samples_to_generate, &found_flag, &randomness_source.uniform_generator, randomness_source.normal_rng_vec.data(), best_points_to_sample);
      }

      status["lhc_" + domain_name + "_domain_found_update"] = found_flag;
      break;
    }  // end case kNull optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimization_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimization_parameters.attr("optimizer_parameters"));
      int num_random_samples = boost::python::extract<int>(optimization_parameters.attr("num_random_samples"));

      if (num_samples_to_generate == 1) {
        ComputeOptimalPointToSampleWithRandomStarts(gaussian_process, gradient_descent_parameters, domain_object, input_container.points_to_sample.data(), input_container.num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, &randomness_source.uniform_generator, randomness_source.normal_rng_vec.data(), best_points_to_sample);
      } else {
        bool random_search_only = false;
        ComputeOptimalSetOfPointsToSample(gaussian_process, gradient_descent_parameters, domain_object, input_container.points_to_sample.data(), input_container.num_to_sample, best_so_far, max_int_steps, max_num_threads, random_search_only, num_random_samples, num_samples_to_generate, &found_flag, &randomness_source.uniform_generator, randomness_source.normal_rng_vec.data(), best_points_to_sample);
      }

      status["gradient_descent_" + domain_name + "_domain_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent optimizer_type
    default: {
      std::fill(best_points_to_sample, best_points_to_sample + input_container.dim*num_samples_to_generate, 0.0);
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid optimizer choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over optimizer_type
}

}  // end unnamed namespace

boost::python::list multistart_expected_improvement_optimization_wrapper(const boost::python::object& optimization_parameters, const GaussianProcess& gaussian_process, const boost::python::list& domain, const boost::python::list& points_to_sample, int num_to_sample, int num_samples_to_generate, double best_so_far, int max_int_steps, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
  // TODO(eliu): (#55793) make domain objects constructible from python; and pass them in through
  // the optimization_parameters python object

  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);
  std::vector<ClosedInterval> domain_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain, input_container.dim, domain_C);

  std::vector<double> best_points_to_sample_C(input_container.dim*num_samples_to_generate);

  DomainTypes domain_type = boost::python::extract<DomainTypes>(optimization_parameters.attr("domain_type"));
  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimization_parameters.attr("optimizer_type"));
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      TensorProductDomain domain_object(domain_C.data(), input_container.dim);
      std::string domain_name("tensor");

      DispatchExpectedImprovementOptimization(optimization_parameters, gaussian_process, input_container, domain_object, domain_name, optimizer_type, num_samples_to_generate, best_so_far, max_int_steps, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kTensorProduct
    case DomainTypes::kSimplex: {
      SimplexIntersectTensorProductDomain domain_object(domain_C.data(), input_container.dim);
      std::string domain_name("simplex");

      DispatchExpectedImprovementOptimization(optimization_parameters, gaussian_process, input_container, domain_object, domain_name, optimizer_type, num_samples_to_generate, best_so_far, max_int_steps, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kSimplex
    default: {
      std::fill(best_points_to_sample_C.begin(), best_points_to_sample_C.end(), 0.0);
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid domain choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over domain_type

  return VectorToPylist(best_points_to_sample_C);
}

boost::python::list evaluate_EI_at_point_list_wrapper(const GaussianProcess& gaussian_process, const boost::python::list& initial_guesses, const boost::python::list& points_to_sample, int num_multistarts, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  PythonInterfaceInputContainer input_container(points_to_sample, gaussian_process.dim(), num_to_sample);
  std::vector<double> result_point_C(input_container.dim);  // not used
  std::vector<double> result_function_values_C(num_multistarts);
  std::vector<double> initial_guesses_C(input_container.dim * num_multistarts);

  CopyPylistToVector(initial_guesses, input_container.dim * num_multistarts, initial_guesses_C);

  bool found_flag = false;
  TensorProductDomain dummy_domain(nullptr, 0);
  EvaluateEIAtPointList(gaussian_process, dummy_domain, initial_guesses_C.data(), input_container.points_to_sample.data(), num_multistarts, input_container.num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, randomness_source.normal_rng_vec.data(), result_function_values_C.data(), result_point_C.data());

  status["evaluate_EI_at_point_list"] = found_flag;

  return VectorToPylist(result_function_values_C);
}

double compute_log_likelihood_wrapper(const boost::python::list& points_sampled, const boost::python::list& points_sampled_value, int dim, int num_sampled, LogLikelihoodTypes objective_type, const boost::python::list& hyperparameters, const boost::python::list& noise_variance) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled, points_sampled_value, noise_variance, points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha, input_container.lengths.data());
  switch (objective_type) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_marginal_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      LogMarginalLikelihoodState log_marginal_state(log_marginal_eval, square_exponential);

      double log_likelihood = log_marginal_eval.ComputeLogLikelihood(log_marginal_state);
      return log_likelihood;
    }  // end case LogLikelihoodTypes::kLogMarginalLikelihood
    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator leave_one_out_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      LeaveOneOutLogLikelihoodState leave_one_out_state(leave_one_out_eval, square_exponential);

      double loo_likelihood = leave_one_out_eval.ComputeLogLikelihood(leave_one_out_state);
      return loo_likelihood;
    }
    default: {
      double log_likelihood = -std::numeric_limits<double>::max();
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid objective mode choice. Setting log likelihood to -DBL_MAX.");
      return log_likelihood;
    }
  }  // end switch over objective_type
}

boost::python::list compute_hyperparameter_grad_log_likelihood_wrapper(const boost::python::list& points_sampled, const boost::python::list& points_sampled_value, int dim, int num_sampled, LogLikelihoodTypes objective_type, const boost::python::list& hyperparameters, const boost::python::list& noise_variance) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled, points_sampled_value, noise_variance, points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha, input_container.lengths.data());
  std::vector<double> grad_log_likelihood(square_exponential.GetNumberOfHyperparameters());
  switch (objective_type) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_marginal_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      LogMarginalLikelihoodState log_marginal_state(log_marginal_eval, square_exponential);

      log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state, grad_log_likelihood.data());
      break;
    }  // end case LogLikelihoodTypes::kLogMarginalLikelihood
    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator leave_one_out_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      LeaveOneOutLogLikelihoodState leave_one_out_state(leave_one_out_eval, square_exponential);

      leave_one_out_eval.ComputeGradLogLikelihood(&leave_one_out_state, grad_log_likelihood.data());
      break;
    }
    default: {
      std::fill(grad_log_likelihood.begin(), grad_log_likelihood.end(), std::numeric_limits<double>::max());
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid objective mode choice. Setting all gradients to DBL_MAX.");
      break;
    }
  }  // end switch over objective_type

  return VectorToPylist(grad_log_likelihood);
}

namespace {

/*
  Utility that dispatches log likelihood optimization (wrt hyperparameters) based on optimizer type.
  This is just used to reduce copy-pasted code.

  Let n_hyper = covariance.GetNumberOfHyperparameters();

  INPUTS:
  optimization_parameters: round_generation/MOE_driver.MOERunner.HyperparameterOptimizationParameters
      Python object containing the LogLikelihoodTypes objective_type and OptimizerTypes optimzer_typ
      to use as well as appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_hyperparameter_optimization_wrapper
  log_likelihood_eval: object supporting evaluation of log likelihood
  covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
  hyperparameter_domain[2][n_hyper]: matrix specifying the boundaries of a n_hyper-dimensional tensor-product
                      domain.  Specified as a list of [x_i_min, x_i_max] pairs, i = 0 .. dim-1
                      Specify in LOG-10 SPACE!
  optimizer_type: type of optimization to use (e.g., null, gradient descent)
  max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
  randomness_source: object containing randomness sources for generating random points in the domain
  status: pydict object; cannot be None

  OUTPUTS:
  randomness_source: PRNG internal states modified
  status: modified on exit to describe whether convergence occurred
  new_hyperparameters[n_hyper]: new hyperparameters found by optimizer to maximize the specified log likelihood measure
*/
template <typename LogLikelihoodEvaluator>
void DispatchHyperparameterOptimization(const boost::python::object& optimization_parameters, const LogLikelihoodEvaluator& log_likelihood_eval, const CovarianceInterface& covariance, ClosedInterval const * restrict hyperparameter_domain, OptimizerTypes optimizer_type, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status, double * restrict new_hyperparameters) {
  bool found_flag = false;
  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      // found_flag set to true; 'dumb' search cannot fail
      // TODO(eliu): REMOVE this assumption and have 'dumb' search function pass
      // out found_flag like every other optimizer does!
      found_flag = true;

      // optimization_parameters must contain an int num_random_samples field, extract it
      int num_random_samples = boost::python::extract<int>(optimization_parameters.attr("num_random_samples"));
      LatinHypercubeSearchHyperparameterOptimization(log_likelihood_eval, covariance, hyperparameter_domain, num_random_samples, max_num_threads, &randomness_source.uniform_generator, new_hyperparameters);
      status["lhc_found_update"] = found_flag;
      break;
    }  // end case kNull for optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimization_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimization_parameters.attr("optimizer_parameters"));
      MultistartGradientDescentHyperparameterOptimization(log_likelihood_eval, covariance, gradient_descent_parameters, hyperparameter_domain, max_num_threads, &found_flag, &randomness_source.uniform_generator, new_hyperparameters);
      status["gradient_descent_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent for optimizer_type
    case OptimizerTypes::kNewton: {
      // optimization_parameters must contain a optimizer_parameters field
      // of type NewtonParameters. extract it
      const NewtonParameters& newton_parameters = boost::python::extract<NewtonParameters&>(optimization_parameters.attr("optimizer_parameters"));
      MultistartNewtonHyperparameterOptimization(log_likelihood_eval, covariance, newton_parameters, hyperparameter_domain, max_num_threads, &found_flag, &randomness_source.uniform_generator, new_hyperparameters);
      status["newton_found_update"] = found_flag;
      break;
    }  // end case kNewton for optimizer_type
    default: {
      std::fill(new_hyperparameters, new_hyperparameters + covariance.GetNumberOfHyperparameters(), 1.0);
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid optimizer choice. Setting all hyperparameters to 1.0.");
      break;
    }
  }  // end switch over optimzer_type for LogLikelihoodTypes::kLogMarginalLikelihood
}

}  // end unnamed namespace

boost::python::list multistart_hyperparameter_optimization_wrapper(const boost::python::object& optimization_parameters, const boost::python::list& hyperparameter_domain, const boost::python::list& points_sampled, const boost::python::list& points_sampled_value, int dim, int num_sampled, const boost::python::list& hyperparameters, const boost::python::list& noise_variance, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled, points_sampled_value, noise_variance, points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha, input_container.lengths.data());
  int num_hyperparameters = square_exponential.GetNumberOfHyperparameters();
  std::vector<double> new_hyperparameters(num_hyperparameters);

  std::vector<ClosedInterval> hyperparameter_domain_C(num_hyperparameters);
  CopyPylistToClosedIntervalVector(hyperparameter_domain, num_hyperparameters, hyperparameter_domain_C);

  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimization_parameters.attr("optimizer_type"));
  LogLikelihoodTypes objective_type = boost::python::extract<LogLikelihoodTypes>(optimization_parameters.attr("objective_type"));
  switch (objective_type) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_likelihood_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);

      DispatchHyperparameterOptimization(optimization_parameters, log_likelihood_eval, square_exponential, hyperparameter_domain_C.data(), optimizer_type, max_num_threads, randomness_source, status, new_hyperparameters.data());
      break;
    }  // end case LogLikelihoodTypes::kLogMarginalLikelihood
    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator log_likelihood_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);

      DispatchHyperparameterOptimization(optimization_parameters, log_likelihood_eval, square_exponential, hyperparameter_domain_C.data(), optimizer_type, max_num_threads, randomness_source, status, new_hyperparameters.data());
      break;
    }  // end case LogLikelihoodTypes::kLeaveOneOutLogLikelihood
    default: {
      std::fill(new_hyperparameters.begin(), new_hyperparameters.end(), 1.0);
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid objective type choice. Setting all hyperparameters to 1.0.");
      break;
    }
  }  // end switch over objective_type

  return VectorToPylist(new_hyperparameters);
}

boost::python::list evaluate_log_likelihood_at_hyperparameter_list_wrapper(const boost::python::list& hyperparameter_list, const boost::python::list& points_sampled, const boost::python::list& points_sampled_value, int dim, int num_sampled, LogLikelihoodTypes objective_mode, const boost::python::list& hyperparameters, const boost::python::list& noise_variance, int num_multistarts, int max_num_threads) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled, points_sampled_value, noise_variance, points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha, input_container.lengths.data());

  std::vector<double> new_hyperparameters_C(square_exponential.GetNumberOfHyperparameters());
  std::vector<double> result_function_values_C(num_multistarts);
  std::vector<double> initial_guesses_C(square_exponential.GetNumberOfHyperparameters() * num_multistarts);

  CopyPylistToVector(hyperparameter_list, square_exponential.GetNumberOfHyperparameters() * num_multistarts, initial_guesses_C);

  TensorProductDomain dummy_domain(nullptr, 0);

  switch (objective_mode) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_likelihood_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      EvaluateLogLikelihoodAtPointList(log_likelihood_eval, square_exponential, dummy_domain, initial_guesses_C.data(), num_multistarts, max_num_threads, result_function_values_C.data(), new_hyperparameters_C.data());
      break;
    }
    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator log_likelihood_eval(input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
      EvaluateLogLikelihoodAtPointList(log_likelihood_eval, square_exponential, dummy_domain, initial_guesses_C.data(), num_multistarts, max_num_threads, result_function_values_C.data(), new_hyperparameters_C.data());
      break;
    }
    default: {
      std::fill(result_function_values_C.begin(), result_function_values_C.end(), -std::numeric_limits<double>::max());
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid objective mode choice. Setting all results to -DBL_MAX.");
      break;
    }
  }

  return VectorToPylist(result_function_values_C);
}

int run_cpp_tests_wrapper() {
  int total_errors = 0;
  int error = 0;

  error = RunLinearAlgebraTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("linear algebra tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("linear algebra tests\n");
  }
  total_errors += error;

  error = RunCovarianceTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("covariance ping tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("covariance ping tests\n");
  }
  total_errors += error;

  error = RunGPPingTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("GP (mean, var, EI) tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("GP (mean, var, EI) tests\n");
  }
  total_errors += error;

  error = RunEIConsistencyTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic, MC EI do not match for 1 potential sample case\n");
  } else {
    OL_SUCCESS_PRINTF("analytic, MC EI match for 1 potential sample case\n");
  }
  total_errors += error;

  error = RunLogLikelihoodPingTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("LogLikelihood ping tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("LogLikelihood ping tests\n");
  }
  total_errors += error;

  error = RunRandomPointGeneratorTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("various random sampling\n");
  } else {
    OL_SUCCESS_PRINTF("various random sampling\n");
  }
  total_errors += error;

  error = RandomNumberGeneratorContainerTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("random number generator containers\n");
  } else {
    OL_SUCCESS_PRINTF("random number generator containers\n");
  }
  total_errors += error;

  error = DomainTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("domain classes\n");
  } else {
    OL_SUCCESS_PRINTF("domain classes\n");
  }
  total_errors += error;

  error = ClosedIntervalTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("ClosedInterval member functions\n");
  } else {
    OL_SUCCESS_PRINTF("ClosedInterval member functions\n");
  }
  total_errors += error;

  error = GeometryToolsTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("geometry tools\n");
  } else {
    OL_SUCCESS_PRINTF("geometry tools\n");
  }
  total_errors += error;

  error = RunOptimizationTests(OptimizerTypes::kGradientDescent);
  if (error != 0) {
    OL_FAILURE_PRINTF("quadratic mock gradient descent optimization\n");
  } else {
    OL_SUCCESS_PRINTF("quadratic mock gradient descent optimization\n");
  }
  total_errors += error;

  error = RunOptimizationTests(OptimizerTypes::kNewton);
  if (error != 0) {
    OL_FAILURE_PRINTF("quadratic mock newton optimization\n");
  } else {
    OL_SUCCESS_PRINTF("quadratic mock newton optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLeaveOneOutLogLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("LOO likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("LOO likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kNewton, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter newton optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter newton optimization\n");
  }
  total_errors += error;

  error = EvaluateLogLikelihoodAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood evaluation at point list\n");
  }
  total_errors += error;

  error = EvaluateEIAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("EI evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("EI evaluation at point list\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationMultipleSamplesTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic simplex EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo simplex EI optimization\n");
  }
  total_errors += error;

  return total_errors;
}

namespace {

/*
  Surrogate "constructor" for GaussianProcess intended only for use by boost::python.  This aliases the normal C++ constructor,
  replacing "double const * restrict" arguments with "const boost::python::list&" arguments.
*/
GaussianProcess * make_gaussian_process(const boost::python::list& hyperparameters, const boost::python::list& points_sampled, const boost::python::list& points_sampled_value, const boost::python::list& noise_variance, int dim, int num_sampled) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(hyperparameters, points_sampled, points_sampled_value, noise_variance, points_to_sample_dummy, dim, num_sampled, num_to_sample);

  SquareExponential square_exponential(input_container.dim, input_container.alpha, input_container.lengths.data());

  return new GaussianProcess(square_exponential, input_container.points_sampled.data(), input_container.points_sampled_value.data(), input_container.noise_variance.data(), input_container.dim, input_container.num_sampled);
}

}  // end unnamed namespace

}  // end namespace optimal_learning

namespace {  // unnamed namespace for exception translation (for BOOST_PYTHON_MODULE(GPP))

/*
  Helper function to build a Python exception type object called "name" within the specified scope.
  The new Python exception will subclass "Exception".

  Afterward, "scope" will have a new callable object called "name" that can be used to
  construct an exception instance. For example, if name = "MyPyException", then in Python:
  >>> import scope
  >>> raise scope.MyPyException("my message")

  WARNING: ONLY call this function from within a BOOST_PYTHON_MODULE block since it only has
  meaning during module construction. After module construction, scope has no meaning and it
  will probably be a dangling pointer or NoneType, leading to an exception or segfault.

  INPUTS:
  name[]: ptr to char array containing the desired name of the new Python exception type
  docstring[]: ptr to char array with the docstring for the new Python exception type
  scope[1]: the scope to add the new exception types to
  OUTPUTS:
  scope[1]: the input scope with the new exception types added
  RETURNS:
  PyObject pointer to the (callable) type object (the new exception type) that was created
*/
OL_WARN_UNUSED_RESULT PyObject * CreatePyExceptionClass(const char * name, const char * docstring, boost::python::scope * scope) {
  std::string scope_name = boost::python::extract<std::string>(scope->attr("__name__"));
  std::string qualified_name = scope_name + "." + name;

  /*
    QUESTION FOR REVIEWER: the docs for PyErr_NewException:
    http://docs.python.org/3/c-api/exceptions.html
    claim that it returns a "new reference."
    The meaning of that phrase:
    http://docs.python.org/release/3.3.3/c-api/intro.html#objects-types-and-reference-counts
    First para: "When a function passes ownership of a reference on to its caller, the caller is said to receive a new reference"

    BUT in this example:
    http://docs.python.org/3.3/extending/extending.html#intermezzo-errors-and-exceptions
    (scroll down to the first code block, reading "Note also that the SpamError variable retains...")
    They set:
      SpamError = PyErr_NewException(...);  (1)
      Py_INCREF(SpamError);                 (2)

    So PyErr_NewException returns a *new reference* that SpamError owns in (1). Then in (2), SpamError owns
    the reference... again?!  (This example appears unchanged since somewhere in python 1.x; maybe it's just old.)
    It seems like this Py_INCREF call is just a 'safety buffer'?

    The doc language (new reference) would lead me to believe that SpamError owns a reference to the new type object.
    When SpamError is done, it should be Py_DECREF'd. (Unless SpamError is never done--which is appropriate here, I
    believe, since type objects should not be deallocated as I have no control of whether all referrers have
    been destroyed.)

    So... is my way/interpretation right? Or should I follow python's example?

    This is not a critical detail. Having too many INCREFs just makes the thing "more" immortal.
    Still I'd like to square this away correctly.
  */
  // const_cast: PyErr_NewExceptionWithDoc expects char *, not char const *. This is an ommission:
  // http://bugs.python.org/issue4949
  // first nullptr: base class for NewException will be PyExc_Exception unless overriden
  // second nullptr: no default member fields (could pass a dict with default fields)
#if defined(PY_MAJOR_VERSION) && PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 7
  PyObject * type_object = PyErr_NewExceptionWithDoc(const_cast<char *>(qualified_name.c_str()), const_cast<char *>(docstring), nullptr, nullptr);
#else
  // PyErr_NewExceptionWithDoc did not exist before Python 2.7, so we "cannot" attach a docstring to our new type object.
  // Attributes of metaclassses (i.e., type objects) are not writeable after creation of that type object. So the only
  // way to add a docstring here would be to build the type from scratch, which is too much pain for just a docstring.
  (void) docstring;  // quiet the compiler warning (unused variable)
  PyObject * type_object = PyErr_NewException(const_cast<char *>(qualified_name.c_str()), nullptr, nullptr);
#endif
  if (!type_object) {
    boost::python::throw_error_already_set();
  }
  scope->attr(name) = boost::python::object(boost::python::handle<>(boost::python::borrowed(type_object)));

  return type_object;
}

/*
  When translating C++ exceptions to Python exceptions, we need to identify a base class. By Python convention,
  we want these to be Python types inheriting from Exception.
  This is a monostate object that knows how to set up these base classes in Python; it also keeps pointers to
  the resulting Python type objects for future use (e.g., instantiating Python exceptions).

  NOTE: this class follows the Monostate pattern, implying GLOBAL STATE.
  http://c2.com/cgi/wiki?MonostatePattern
  http://www.informit.com/guides/content.aspx?g=cplusplus&seqNum=147
  That is, the various PyObject * for Python type objects (and others) are stored as private, static variables.

  We use monostate because once the Python type objects have been created, we need to hold on to references to
  them until the end of time. Type objects must never be deallocated:
  http://docs.python.org/release/3.3.3/c-api/intro.html#objects-types-and-reference-counts
  "The sole exception are the type objects; since these must never be deallocated, they are typically static PyTypeObject objects."
  Also, see python/object.h's header comments (object.h, Python 2.7):
  "Type objects are exceptions to the first rule; the standard types are represented by
  statically initialized type objects, although work on type/class unification
  for Python 2.2 made it possible to have heap-allocated type objects too."

  Thus, as long as the enclosing Python module lives, we need to hold references to the (exception) type objects,
  which (as far as I know) requires global state. So storing the type objects in boost::python::object (which could
  be destructed) is not an option. We could also heap allocate this container and never delete,
  but that seems even more confusing. Besides, static variables is how it is done in Python.

  Additionally, this class protects against redundant calls to PyErr_NewException (which creates exception type objects).
  Redundant here means creating multiple exception types with the same name.  Failing to do so would add MULTIPLE
  instances of the "same" type (Python has no ODR), which is confusing to the user.

  NOTE: we cannot mark PyObject * pointers as pointers to const, e.g.,
  const PyObject * some_object;
  because Python C-API calls will modify the objects. HOWEVER, DO NOT CHANGE these pointers and
  DO NOT CHANGE the things they point to!  (Outside of Python calls that is.)
  The pointers in the monostate are meant to be "conceptually" const.
*/
class PyExceptionClassContainer {
 public:
  /*
    Initializes the (mono-) state. This defines new python type objects (for exceptions) and saves the
    python scope that they are defined in.

    If Initialize() was already called previously (with no subsequent calls to Reset()), this
    function does nothing, preventing defining multiple "identical" types in Python.

    If Initialize() is not called at program start or after Reset(), all newly translated exceptions will
    be of type: default_exception_type_object_.

    WARNING: NOT THREAD SAFE in C++. It might be thread-safe in Python calls; not sure
    how the GIL is handled here.
    However there is no reason to call this from multiple threads in C++ so I'm ignoring the issue.

    INPUTS:
    scope[1]: the scope to add the new exception types to
    OUTPUTS:
    scope[1]: the input scope with the new exception types added
  */
  void Initialize(boost::python::scope * scope) OL_NONNULL_POINTERS {
    // Prevent duplicate definitions (in Python) of the same objects
    if (!initialized_) {
      scope_ = scope;

      // TODO(eliu): If listing exception type objects here gets unwieldly, we can store them in an
      // array<> of typle<>, making Initialize() just a simple for (item : array) { ... } loop.
      static char const * bounds_exception_docstring = "value not in range [min, max].";
      bounds_exception_type_object_ = CreatePyExceptionClass(bounds_exception_name_, bounds_exception_docstring, scope_);

      static char const * invalid_value_exception_docstring = "value != truth (+/- tolerance)";
      invalid_value_exception_type_object_ = CreatePyExceptionClass(invalid_value_exception_name_, invalid_value_exception_docstring, scope_);

      static char const * singular_matrix_exception_docstring = "num_rows X num_cols matrix is singular";
      singular_matrix_exception_type_object_ = CreatePyExceptionClass(singular_matrix_exception_name_, singular_matrix_exception_docstring, scope_);

      initialized_ = true;
    }
  }

  /*
    Reset the state back to default. Afer this call, translated exceptions will be of type
    default_exception_type_object_. Future calls to Initialize() will define new exception
    types in another (not necessarily different) scope.

    This is not recommended.

    WARNING: NOT THREAD SAFE. See Initialize() comments.

    WARNING: This makes the existing PyObjects *unreachable* from C++.
    It is unsafe to DECREF our PyObject pointers; we cannot guarantee that these type
    objects will outlive all instances. (*I think*)
  */
  void Reset() {
    initialized_ = false;
    scope_ = nullptr;
    bounds_exception_type_object_ = default_exception_type_object_;
    invalid_value_exception_type_object_ = default_exception_type_object_;
    singular_matrix_exception_type_object_ = default_exception_type_object_;
  }

  PyObject * bounds_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return bounds_exception_type_object_;
  }

  PyObject * invalid_value_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return invalid_value_exception_type_object_;
  }

  PyObject * singular_matrix_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return singular_matrix_exception_type_object_;
  }

 private:
  // names to use in Python (just for convenience)
  constexpr static char const * const bounds_exception_name_ = optimal_learning::BoundsException<double>::kName;
  constexpr static char const * const invalid_value_exception_name_ = optimal_learning::InvalidValueException<double>::kName;
  constexpr static char const * const singular_matrix_exception_name_ = optimal_learning::SingularMatrixException::kName;

  // Fall back to this type object if something has not been initialized or we are otherwise confused.
  static PyObject * const default_exception_type_object_;

  // pointers to Python callable objects that build the Python exception classes
  static PyObject * bounds_exception_type_object_;
  static PyObject * invalid_value_exception_type_object_;
  static PyObject * singular_matrix_exception_type_object_;

  // scope that these exception objects will live in
  static boost::python::scope * scope_;

  // whether this class has been properly initialized
  static bool initialized_;
};

PyObject * const PyExceptionClassContainer::default_exception_type_object_ = PyExc_RuntimeError;
PyObject * PyExceptionClassContainer::bounds_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
PyObject * PyExceptionClassContainer::invalid_value_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
PyObject * PyExceptionClassContainer::singular_matrix_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
boost::python::scope * PyExceptionClassContainer::scope_ = nullptr;
bool PyExceptionClassContainer::initialized_ = false;

/*
  Translate optimal_learning::BoundsException to a Python exception, maintaining the data fields.

  INPUTS:
  except: C++ exception to translate
  py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
                             translated Python exception base class. We assume that this type inherits from Exception.
  RETURNS:
  **NEVER RETURNS**
*/
template <typename ValueType>
OL_NORETURN void TranslateBoundsException(const optimal_learning::BoundsException<ValueType>& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.bounds_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("value") = except.value();
  instance.attr("min") = except.min();
  instance.attr("max") = except.max();

  // Note: SetObject gets ownership (*not* borrow) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*
  Translate optimal_learning::InvalidValueException to a Python exception, maintaining the data fields.

  INPUTS:
  except: C++ exception to translate
  py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
                             translated Python exception base class. We assume that this type inherits from Exception.
  RETURNS:
  **NEVER RETURNS**
*/
template <typename ValueType>
OL_NORETURN void TranslateInvalidValueException(const optimal_learning::InvalidValueException<ValueType>& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.invalid_value_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("value") = except.value();
  instance.attr("truth") = except.truth();
  instance.attr("tolerance") = except.tolerance();

  // Note: SetObject gets ownership (*not* borrow) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*
  Translate optimal_learning::SingularMatrixException to a Python exception, maintaining the data fields.

  INPUTS:
  except: C++ exception to translate
  py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
                             translated Python exception base class. We assume that this type inherits from Exception.
  RETURNS:
  **NEVER RETURNS**
*/
OL_NORETURN void TranslateSingularMatrixException(const optimal_learning::SingularMatrixException& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.singular_matrix_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("num_rows") = except.num_rows();
  instance.attr("num_cols") = except.num_cols();
  // TODO(eliu): (#?????) this would make more sense as a numpy array/matrix
  instance.attr("matrix") = VectorToPylist(except.matrix());

  // Note: SetObject gets ownership (*not* borrow/steal) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*
  Register an exception translator (C++ to Python) with boost python for ExceptionType, using the callable translate.
  Boost python expects only unary exception translators (w/the exception to translate as the argument), so we use
  a lambda to capture additional arguments for our translators.

  TEMPLATE PARAMETERS:
  ExceptionType: the type of the exception that the user wants to register
  Translator: a Copyconstructible type such that the following code is well-formed:
      void SomeFunc(ExceptionType except, const PyExceptionClassContainer& py_exception_type_objects) {
        translate(except, py_exception_type_objects);
      }
    Currently, the use cases in BOOST_PYTHON_MODULE(GPP) pass translate as a function pointer.
    This follows the requirements for boost::python::register_exception_translator:
    http://www.boost.org/doc/libs/1_55_0/libs/python/doc/v2/exception_translator.html
    http://en.cppreference.com/w/cpp/concept/CopyConstructible
  INPUTS:
  py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
                             translated Python exception base class. We assume that this type inherits from Exception.
  translate: an instance of Translator satisfying the requirements described above in TEMPLATE PARAMETERS.
*/
template <typename ExceptionType, typename Translator>
void RegisterExceptionTranslatorWithPayload(const PyExceptionClassContainer& py_exception_type_objects, Translator translate) {
  static_assert(std::is_copy_constructible<Translator>::value, "Exception translator must be copy constructible.");
  // lambda capturing the closure of translate (py_exception_type_objects)
  // py_exception_type_objects captured by value; we don't want a dangling reference
  // translate captured by value; we know it is copyconstructible
  auto translate_exception =
      [py_exception_type_objects, translate](const ExceptionType& except) {
    translate(except, py_exception_type_objects);
  };
  // TODO(eliu): if/when template'd lambdas become available (C++14?), we can kill this function.

  // nullptr suppresses a superfluous compiler warning b/c boost::python::register_exception_translator
  // defaults a (dummy) pointer argument to 0.
  boost::python::register_exception_translator<ExceptionType>(translate_exception, nullptr);
}

/*
  Helper that registers exception translators to convert C++ exceptions to Python exceptions.
  This is just a convenient place to register translators for optimal_learning's exceptions.

  NOTE: PyExceptionClassContainer (monostate class) must be properly initialized first!
  Otherwise all translators will translate to PyExceptionClassContainer::default_exception_type_object_
  (e.g., PyExc_RuntimeError).
*/
void RegisterOptimalLearningExceptions() {
  PyExceptionClassContainer py_exception_type_objects;

  // Note: boost python stores exception translators in a LIFO stack. The most recently (in code execution order)
  // registered translator gets "first shot" at matching incoming exceptions. Reference:
  // http://www.boost.org/doc/libs/1_55_0/libs/python/doc/v2/exception_translator.html
  RegisterExceptionTranslatorWithPayload<optimal_learning::SingularMatrixException>(py_exception_type_objects, &TranslateSingularMatrixException);
  RegisterExceptionTranslatorWithPayload<optimal_learning::InvalidValueException<int>>(py_exception_type_objects, &TranslateInvalidValueException<int>);
  RegisterExceptionTranslatorWithPayload<optimal_learning::InvalidValueException<double>>(py_exception_type_objects, &TranslateInvalidValueException<double>);
  RegisterExceptionTranslatorWithPayload<optimal_learning::BoundsException<int>>(py_exception_type_objects, &TranslateBoundsException<int>);
  RegisterExceptionTranslatorWithPayload<optimal_learning::BoundsException<double>>(py_exception_type_objects, &TranslateBoundsException<double>);
}

}  // end unnamed namespace

namespace {  // unnamed namespace for BOOST_PYTHON_MODULE(GPP) definition

// TODO(eliu): (#59677) improve docstrings for the GPP module and for the classes, functions, etc
//   in it exposed to Python. Many of them are a bit barebones at the moment.
BOOST_PYTHON_MODULE(GPP) {
  using namespace boost::python;  // NOLINT(build/namespaces) irrelevant b/c it's inside an unnamed namespace

  scope current_scope;

  // initialize PyExceptionClassContainer monostate class and set its scope to this module (GPP)
  PyExceptionClassContainer py_exception_type_objects;
  py_exception_type_objects.Initialize(&current_scope);

  // Register exception translators to convert C++ exceptions to python exceptions.
  // Note: if adding additional translators, recall that boost python maintains translators in a LIFO
  // stack. See RegisterOptimalLearningExceptions() docs for more details.
  RegisterOptimalLearningExceptions();

  bool show_user_defined = true;
  bool show_py_signatures = true;
  bool show_cpp_signatures = true;
  // enable full docstrings for the functions, enums, ctors, etc. provided in this module.
  boost::python::docstring_options doc_options(show_user_defined, show_py_signatures, show_cpp_signatures);

  current_scope.attr("__doc__") = R"%%(
    This module is the python interface to the C++ component of the Metrics Optimization Engine, or MOE. It exposes
    enumerated types for specifying interface behaviors, C++ objects for computation and communication, as well as
    various functions for optimization and testing/data exploration.

    OVERVIEW:
    TODO(eliu): when we come up with a "README" type overview for MOE, that or parts of that should be incorporated here.

    MOE is a black-box global optimization method for objectives (e.g., click-through rate, delivery time, happiness)
    that are time-consuming/expensive to measure, highly complex, non-convex, nontrivial to predict, or all of the above.
    It optimizes the user-chosen objective with respect to the user-specified parameters; e.g., scoring weights,
    thresholds, learning rates, etc.

    MOE examines the history of all experiments (tested parameters, measured metrics, measurement noise) to date and
    outputs the next set of parameters to sample that it believes will produce the best results.

    To perform this optimization, MOE models the world with a Gaussian Process (GP). Conceptually, this means that MOE
    assumes the performance of every set of parameters (e.g., weights) is governed by a Gaussian with some mean and
    variance. The mean/variance are functions of the historical data. Compared to running live experiments,
    computations on a GP are cheap. So MOE uses GPs to predict and maximize the Expected Improvement, producing
    a new set of experiment parameters that produces the greatest (expected) improvement over the best performing
    parameters from the historical data.

    Thus, using MOE breaks down into three main steps:
    1) Model Selection/Hyperparameter Optimization: the GP (through a "covariance" function) has several hyperparameters
       that are not informed by the model. We first need to choose an appropriate set of hyperparameters. For example,
       we could choose the hyperparameters that maximize the likelihood that the model produced the observed data.
       multistart_hyperparameter_optimization() is the primary endpoint for this functionality.

    2) Construct the Gaussian Process: from the historical data and the hyperparameters chosen in step 1, we can build
       a GP that MOE will use as a proxy for the behavior of the objective in the real world. In this sense, the GP's
       predictions are a type of regression.
       GaussianProcess() constructs a GP.

    3) Select new Experiment Parameters: with the GP from step 2, MOE now has a model for the "real world." Using this
       GP model, we will select the next experiment parameters (or set of parameters) for live measurement. These
       new parameters are the ones MOE thinks will produce the biggest gain over the historical best.
       multistart_expected_improvement_optimization() is the primary endpoint for this functionality.

    DETAILS:
    For more information, consult the docstrings for the entities exposed in this module.  (Everybody has one!)
    These docstrings currently provide fairly high level (and sometimes sparse) descriptions.
    For further details, see the file documents for the C++ hpp and cpp files. Header (hpp) files contain more high
    level descriptions/motivation whereas source (cpp) files contain more [mathematical] details.
    gpp_math.hpp and gpp_model_selection_and_hyperparameter_optimization.hpp are good starting points for more reading.
    TODO(eliu): when we have gemdoc or whatever, point this to those docs as well.

    Now we will provide an overview of the enums, classes, and endpoints provided in this module.
    Note: Each entity provided in this module has a docstring; this is only meant to be an overview. Consult the
          individual docstrings for more information and/or see the C++ docs.

    Exceptions:
      We expose equivalent Python definitions for the exception classes in gpp_exception.hpp. We also provide
      translators so that C++ exceptions will be caught and rethrown as their Python counterparts. The type objects
      (e.g., BoundsException) are referenced by module-scope names in Python.

    Enums:
      We provide some enum types defined in C++. Values from these enum types are used to signal information about
      which domain, which optimizer, which log likelihood objective, etc. to use throughout this module. We define the
      enums in C++ because not all currently in-use version of Python support enums natively. Additionally, we wanted
      the strong typing. In particular, a function expecting a DomainTypes enum *cannot* take (int) 0 as an input
      even if kTensorProduct maps to the value 0. In general, *never* rely on the particular value of each enum name.
        <> DomainTypes, LogLikelihoodTypes, OptimizerTypes

    Objects:
      We currently expose constructors for a few C++ objects:
        <> GaussianProcess: for one set of historical data (and hyperparameters), this represents the GP model. It is fairly
           expensive to create, so the intention is that users create it once and pass it to all functions in this module
           that need it (as opposed to recreating it every time).  Constructing the GP is noted as step 2 of MOE above.
        <> GradientDescentParameters, NewtonParameters: structs that hold tolerances, max step counts, learning rates, etc.
           that control the behavior of the derivative-based optimizers
        <> RandomnessSourceContainer: container for a uniform RNG and a normal (gaussian) RNG. These are needed by the C++ to
           guarantee that multi-threaded runs see different (and consistent) randomness. This class also exposes several
           functions for setting thread-safe seeds (both explicitly and automatically).

    Optimization:
      We expose two main optimization routines. One for model selection and the other for experimental cohort selection.
      These routines provide multistart optimization (with choosable optimizer, domain, and objective type) and are
      the endpoints to use for steps 1 and 3 of MOE dicussed above.
      These routines are multithreaded.
        <> multistart_hyperparameter_optimization: optimize the specified log likelihood measure to yield the hyperparameters
           that produce the "best" model.
        <> multistart_expected_improvement_optimization: optimize the expected improvement to yield the experimental parameters
           that the user should test next.

    Testing/Exploring:
      These endpoints are provided for testing core C++ functionality as well as exploring the behaviors of
      Gaussian Processes, Expected Improvement, etc.  GP and EI functions require a GaussianProcess object.
      <> Log Likelihood (model fit): compute_log_likelihood, compute_hyperparameter_grad_log_likelihood
      <> GPs: get_mean, get_grad_mean, get_var, get_chol_var, get_grad_var
              Note: get_grad_var is the gradient of get_chol_var; the gradient of get_var is not currently provided.
      <> EI: compute_expected_improvement, compute_grad_expected_improvement

    Plotting:
      These endpoints are useful for plotting log likelihood or expected improvement (or other applications needing
      lists of function values).
      These routines are multithreaded.
      <> evaluate_log_likelihood_at_hyperparameter_list: compute selected log likelihood measures for each set of specified
           hyperparameters. Equivalent to but much faster than calling compute_log_likelihood() in a loop.
      <> evaluate_EI_at_point_list: compute expected improvement for each point (cohort parameters) in an input list.
           Equivalent to but much faster than calling compute_expected_improvement in a loop.
    )%%";

  // enum for optimizers, domains, and log likelihood-type objective functions
  // see top of this file for C++ declarations
  enum_<optimal_learning::OptimizerTypes>("OptimizerTypes", R"%%(
    C++ enums to describe the available optimizers:
    kNull: null optimizer (use for 'dumb' search)
    kGradientDescent: gradient descent
    kNewton: Newton's Method
      )%%")
      .value("null", optimal_learning::OptimizerTypes::kNull)
      .value("gradient_descent", optimal_learning::OptimizerTypes::kGradientDescent)
      .value("newton", optimal_learning::OptimizerTypes::kNewton)
      ;  // NOLINT, this is boost style

  enum_<optimal_learning::DomainTypes>("DomainTypes", R"%%(
    C++ enums to describe the available domains:
    kTensorProduct: a d-dimensional tensor product domain, D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]
    kSimplex: intersection of kTensorProduct with the unit d-simplex

    The unit d-simplex is defined as the set of x_i such that:
    1) x_i >= 0 \forall i  (i ranging over dimension)
    2) \sum_i x_i <= 1
    (Constrained) optimization is performed over domains.
      )%%")
      .value("tensor_product", optimal_learning::DomainTypes::kTensorProduct)
      .value("simplex", optimal_learning::DomainTypes::kSimplex)
      ;  // NOLINT, this is boost style

  enum_<optimal_learning::LogLikelihoodTypes>("LogLikelihoodTypes", R"%%(
    C++ enums to describe the available log likelihood-like measures of model fit:
    kLogMarginalLikelihood: the probability of the observations given [the assumptions of] the model
    kLeaveOneOutLogLikelihood: cross-validation based measure, this indicates how well
      the model explains itself by computing successive log likelihoods, leaving one
      training point out each time.
      )%%")
      .value("log_marginal_likelihood", optimal_learning::LogLikelihoodTypes::kLogMarginalLikelihood)
      .value("leave_one_out_log_likelihood", optimal_learning::LogLikelihoodTypes::kLeaveOneOutLogLikelihood)
      ;  // NOLINT, this is boost style

  /*
    NOTE: ALL ARRAYS/LISTS MUST BE FLATTENED!
    What that means:
    Matrices will be described as A[dim1][dim2]...[dimN]
    To FLATTEN a matrix is to lay it out in memory C-style;
    i.e., rightmost index varies the most rapidly.
    For example: A[3][4] =
    [4  32  5  2
    53 12  8  1
    81  2  93 0]
    would be FLATTENED into an array:
    A_flat[12] = [4 32 5 2 53 12 8 1 81 2 93 0]

    Details on "pylist hyperparameters[2]:"
    This will be a python list such that:
    hyperparameters[0] = double precision number: \alpha (=\sigma_f^2, signal variance)
    hyperparameters[1] = pylist lengths[dim]: list of length scales for covariance (doubles)
    For example, if dim = 3, you might set in Python:
    hyperparameters_for_C = [2.1, [1.2, 3.1, 0.4]]
    for \alpha = 2.1, and for length scales 1.2, 3.1, 0.4

    NOTE2: Below, we use raw strings (C++11) to pass multiline string
    literals to boost for python docstrings. Our delimiter is: %%. The
    format is: R"%%(put anything here, no need to escape chars)%%"
  */

  def("compute_expected_improvement", optimal_learning::compute_expected_improvement_wrapper, R"%%(
    Compute expected improvement.
    If num_to_sample is small enough (= 1) AND force_monte_carlo is false, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based EI computation is used.

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample
    int max_int_steps: number of MC integration points in EI
    double best_so_far: best known value of objective so far
    bool force_monte_carlo: true to force monte carlo evaluation of EI
    RandomnessSourceContainer randomness_source: object containing randomness sources; only thread 0's source is used

    RETURNS:
    double result: computed EI
    )%%");

  def("compute_grad_expected_improvement", optimal_learning::compute_grad_expected_improvement_wrapper, R"%%(
    Compute the gradient of expected improvement evaluated at current_point.
    If num_to_sample is small enough (= 1) AND force_monte_carlo is false, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based EI computation is used.

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample
    int max_int_steps: number of MC integration points in EI
    double best_so_far: best known value of objective so far
    bool force_monte_carlo: true to force monte carlo evaluation of EI
    RandomnessSourceContainer randomness_source: object containing randomness sources; only thread 0's source is used
    pylist current_point[dim]: current point being considered

    RETURNS:
    pylist result[dim]: gradient of EI (computed at current_point)
    )%%");

  def("get_mean", optimal_learning::get_mean_wrapper, R"%%(
    Compute the (predicted) mean, mus, of the Gaussian Process posterior.
    mus_i = Ks_{i,k} * K^-1_{k,l} * y_l

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample

    RETURNS:
    pylist result[num_to_sample]: mean of points to be sampled
    )%%");

  def("get_grad_mean", optimal_learning::get_grad_mean_wrapper, R"%%(
    Compute the gradient of the (predicted) mean, mus, of the Gaussian Process posterior.
    Gradient is computed wrt each point in points_to_sample.
    Known zero terms are dropped (see below).
    mus_i = Ks_{i,k} * K^-1_{k,l} * y_l
    In principle, we compute \pderiv{mus_i}{Xs_{d,p}}
    But this is zero unless p == i. So this method only returns the "block diagonal."

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample

    RETURNS:
    pylist result[dim*num_to_sample]: gradient of mean values, ordered in num_to_sample rows of size dim
    )%%");

  def("get_var", optimal_learning::get_var_wrapper, R"%%(
    Compute the (predicted) variance, Vars, of the Gaussian Process posterior.
    L * L^T = K
    V = L^-1 * Ks^T
    Vars = Kss - (V^T * V)
    Expanded index notation:
    Vars_{i,j} = Kss_{i,j} - Ks_{i,l} * K^-1_{l,k} * Ks^T_{k,j}

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample

    RETURNS:
    pylist result[num_to_sample][num_to_sample]:
      matrix of variances, ordered as num_to_sample rows of length num_to_sample
    )%%");

  def("get_chol_var", optimal_learning::get_chol_var_wrapper, R"%%(
    Computes the Cholesky Decomposition of the predicted GP variance:
    L * L^T = Vars, where Vars is the output of get_var().
    See that function's docstring for further details.

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample

    RETURNS:
    pylist result[num_to_sample][num_to_sample]: Cholesky Factorization of the
      matrix of variances, ordered as num_to_sample rows of length num_to_sample
    )%%");

  def("get_grad_var", optimal_learning::get_grad_var_wrapper, R"%%(
    Compute gradient of the Cholesky Factorization of the (predicted) variance, Vars, of the Gaussian Process posterior.
    Gradient is computed wrt the point with index var_of_grad in points_to_sample.
    L * L^T = K
    V = L^-1 * Ks^T
    Vars = Kss - (V^T * V)
    Expanded index notation:
    Vars_{i,j} = Kss_{i,j} - Ks_{i,l} * K^-1_{l,k} * Ks^T_{k,j}
    Then the Cholesky Decomposition is Ls * Ls^T = Vars

    General derivative expression: \pderiv{Ls_{i,j}}{Xs_{d,p}}
    We compute this for p = var_of_grad

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample
    int var_of_grad: dimension to differentiate in

    RETURNS:
    pylist result[num_to_sample][num_to_sample][dim]:
      tensor of cholesky factorized variance gradients, ordered as num_to_sample blocks, each having
      num_to_sample rows with length dim
    )%%");

  def("multistart_expected_improvement_optimization", optimal_learning::multistart_expected_improvement_optimization_wrapper, R"%%(
    Optimize expected improvement over the specified domain using the specified optimization method.
    Can optimize for num_samples_to_generate new points to sample (i.e.,
    experiments to run) simultaneously.

    The ExpectedImprovementOptimizationParameters object is a python class defined in:
    round_generation/MOE_driver.MOERunner.ExpectedImprovementOptimizationParameters
    See that class definition for more details.

    This function expects it to have the fields:
    domain_type (DomainTypes enum from this file)
    optimizer_type (OptimizerTypes enum from this file)
    num_random_samples (int, number of samples to 'dumb' search over, if 'dumb' search is being used.
                              e.g., if optimizer = kNull or if samples_to_generate > 1)
    optimizer_parameters (*Parameters struct (gpp_optimization_parameters.hpp) where * matches optimizer_type
                          unused if optimizer_type == kNull)

    WARNING: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads

    INPUTS:
    ExpectedImprovementOptimizationParameters optimization_parameters:
        python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
        appropriate parameter structs e.g., NewtonParameters for type kNewton)
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist domain[dim][2]: [lower, upper] bound pairs for each dimension
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_to_sample: number of points to sample
    int num_samples_to_generate: how many simultaneous experiments you would like to run
    double best_so_far: best known value of objective so far
    int max_int_steps: number of MC integration points in EI and grad_EI
    int max_num_threads: max number of threads to use during EI optimization
    RandomnessSourceContainer randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    RETURNS:
    pylist result[num_samples_to_generate][dim]: next set of points to eval
    )%%");

  def("evaluate_EI_at_point_list", optimal_learning::evaluate_EI_at_point_list_wrapper, R"%%(
    Evaluates the expected improvement at each point in initial_guesses.
    Useful for plotting.

    Equivalent to
    result = []
    for point in initial_guesses:
        result.append(compute_expected_improvement(point, ...))

    But this method is substantially faster (loop in C++ and multithreaded).

    WARNING: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads

    INPUTS:
    GaussianProcess gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    pylist initial_guesses[num_multistarts][dim]: points at which to evaluate EI
    pylist points_to_sample[num_to_sample][dim]: points to sample
    int num_multistarts: number of locations from which to start gradient descent
    int num_to_sample: number of points to sample
    double best_so_far: best known value of objective so far
    int max_int_steps: number of MC integration points in EI and grad_EI
    int max_num_threads: max number of threads to use during EI optimization
    RandomnessSourceContainer randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    OUTPUTS:
    pylist result[num_multistarts]: EI values at each point of the initial_guesses list, in the same order
    )%%");

  def("compute_log_likelihood", optimal_learning::compute_log_likelihood_wrapper, R"%%(
    Computes the specified log likelihood measure of model fit using the given
    hyperparameters.

    pylist points_sampled[num_sampled][dim]: points already sampled
    pylist points_sampled_value[num_sampled]: objective value at each sampled point
    int dim: dimension of parameter space
    int num_sampled: number of points already sampled
    LogLikelihoodTypes objective_mode: describes which log likelihood measure to compute (e.g., kLogMarginalLikelihood)
    pylist hyperparameters[2]: covariance hyperparameters; see "Details on ..." section at the top of BOOST_PYTHON_MODULE
    pylist noise_variance[num_sampled]: \sigma_n^2, noise variance (one value per sampled point)

    RETURNS:
    double result: computed log marginal likelihood of prior
    )%%");

  def("compute_hyperparameter_grad_log_likelihood", optimal_learning::compute_hyperparameter_grad_log_likelihood_wrapper, R"%%(
    Computes the gradient of the specified log likelihood measure of model fit using the given
    hyperparameters. Gradient computed wrt the given hyperparameters.

    n_hyper denotes the number of hyperparameters.

    pylist points_sampled[num_sampled][dim]: points already sampled
    pylist points_sampled_value[num_sampled]: objective value at each sampled point
    int dim: dimension of parameter space
    int num_sampled: number of points already sampled
    LogLikelihoodTypes objective_mode: describes which log likelihood measure to compute (e.g., kLogMarginalLikelihood)
    pylist hyperparameters[2]: covariance hyperparameters; see "Details on ..." section at the top of BOOST_PYTHON_MODULE
    pylist noise_variance[num_sampled]: \sigma_n^2, noise variance (one value per sampled point)

    RETURNS:
    pylist result[n_hyper]: gradients of log marginal likelihood wrt hyperparameters
    )%%");

  def("multistart_hyperparameter_optimization", optimal_learning::multistart_hyperparameter_optimization_wrapper, R"%%(
    Optimize the specified log likelihood measure over the specified domain using the specified optimization method.

    The HyperparameterOptimizationParameters object is a python class defined in:
    round_generation/MOE_driver.MOERunner.HyperparameterOptimizationParameters
    See that class definition for more details.

    This function expects it to have the fields:
    objective_type (LogLikelihoodTypes enum from this file)
    optimizer_type (OptimizerTypes enum from this file)
    num_random_samples (int, number of samples to 'dumb' search over, only used if optimizer_type == kNull)
    optimizer_parameters (*Parameters struct (gpp_optimization_parameters.hpp) where * matches optimizer_type
                          unused if optimizer_type == kNull)

    n_hyper denotes the number of hyperparameters.

    INPUTS:
    HyperparameterOptimizationParameters optimization_parameters:
        python object containing the LogLikelihoodTypes objective to use, OptimizerTypes optimzer_type
        to use as well as appropriate parameter structs e.g., NewtonParameters for type kNewton
    pylist hyperparameter_domain[2*n_hyper]: [lower, upper] bound pairs for each hyperparameter dimension in LOG-10 SPACE
    pylist points_sampled[num_sampled][dim]: points already sampled
    pylist points_sampled_value[num_sampled]: objective value at each sampled point
    int dim: dimension of parameter space
    int num_sampled: number of points already sampled
    pylist hyperparameters[2]: covariance hyperparameters; see "Details on ..." section at the top of BOOST_PYTHON_MODULE
    pylist noise_variance[num_sampled]: \sigma_n^2, noise variance (one value per sampled point)
    int max_num_threads: max number of threads to use during Newton optimization
    RandomnessSourceContainer randomness_source: object containing randomness source (UniformRandomGenerator) for LHC sampling
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    RETURNS::
    pylist next_hyperparameters[n_hyper]: optimized hyperparameters
    )%%");

  def("evaluate_log_likelihood_at_hyperparameter_list", optimal_learning::evaluate_log_likelihood_at_hyperparameter_list_wrapper, R"%%(
    Evaluates the specified log likelihood measure of model fit at each member of
    hyperparameter_list. Useful for plotting.

    Equivalent to
    result = []
    for hyperparameters in hyperparameter_list:
        result.append(compute_log_likelihood(hyperparameters, ...))

    But this method is substantially faster (loop in C++ and multithreaded).

    n_hyper denotes the number of hyperparameters

    INPUTS:
    pylist hyperparameter_list[num_multistarts][n_hyper]: list of hyperparameters at which to evaluate log likelihood
    pylist points_sampled[num_sampled][dim]: points already sampled
    pylist points_sampled_value[num_sampled]: objective value at each sampled point
    int dim: dimension of parameter space
    int num_sampled: number of points already sampled
    LogLikelihoodTypes objective_mode: describes which log likelihood measure to compute (e.g., kLogMarginalLikelihood)
    pylist hyperparameters[2]: covariance hyperparameters; see "Details on ..." section at the top of BOOST_PYTHON_MODULE
    pylist noise_variance[num_sampled]: \sigma_n^2, noise variance (one value per sampled point)
    num_multistarts: number of latin hypercube samples to use
    int max_num_threads: max number of threads to use during Newton optimization

    RETURNS:
    pylist result[num_multistarts]: log likelihood values at each point of the hyperparameter_list list, in the same order
    )%%");

  def("run_cpp_tests", optimal_learning::run_cpp_tests_wrapper, R"%%(
    Runs all current C++ unit tests and reports failures.

    RETURNS:
    number of test failures. expected to be 0.
    )%%");

  class_<optimal_learning::GradientDescentParameters, boost::noncopyable>("GradientDescentParameters", init<int, int, int, double, double, double, double>(R"%%(
    Constructor for a GradientDescentParameters object.

    int num_multistarts: number of initial guesses to try in multistarted gradient descent
    int max_num_steps: maximum number of gradient descent iterations
    int max_num_restarts: maximum number of times we are allowed to call gradient descent.  Should be >= 2 as a minimum.
    double gamma: exponent controlling rate of step size decrease (see get_next_step)
    double pre_mult: scaling factor for step size (see get_next_step)
    double max_relative_change: max relative change allowed per iteration of gradient descent
    double tolerance: when the distance moved btwn steps falls below a factor of this value, stop
    )%%"));

  class_<optimal_learning::NewtonParameters, boost::noncopyable>("NewtonParameters", init<int, int, double, double, double, double>(R"%%(
    Constructor for a NewtonParameters object.

    int num_multistarts: number of initial guesses to try in multistarted newton
    int max_num_steps: maximum number of newton iterations
    double gamma: exponent controlling rate of time_factor growth (see NewtonHyperparameterOptimization)
    double time_factor: initial amount of additive diagonal dominance (see NewtonHyperparameterOptimization())
    double max_relative_change: max relative change allowed per iteration of newton (UNUSED)
    double tolerance: when the magnitude of the gradient falls below this value, stop
    )%%"));

  class_<optimal_learning::GaussianProcess, boost::noncopyable>("GaussianProcess", no_init)
      .def("__init__", boost::python::make_constructor(&optimal_learning::make_gaussian_process), R"%%(
    Constructor for a GaussianProcess object

    INPUTS:
    pylist hyperparameters[2]: covariance hyperparameters; see "Details on ..." section at the top of BOOST_PYTHON_MODULE
    pylist points_sampled[dim][num_sampled]: points that have already been sampled
    pylist points_sampled_value[num_sampled]: values of the already-sampled points
    pylist noise_variance[num_sampled]: the \sigma_n^2 (noise variance) associated w/observation, points_sampled_value
    dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    num_sampled: number of already-sampled points
          )%%")
      ;  // NOLINT, this is boost style

  class_<optimal_learning::RandomnessSourceContainer, boost::noncopyable>("RandomnessSourceContainer", init<int>(R"%%(
    Constructor for a RandomnessSourceContainer with enough random sources for at most num_threads simultaneous accesses.

    Random sources are seeded to a repeatable combination of the default seed and the thread id.
    Call SetRandomizedUniformGeneratorSeed() and/or SetRandomizedNormalRNGSeed to use
    an automatically generated (and less repeatable) seed(s).

    INPUTS:
    int num_threads: the max number of threads this object will be used with (sets the number of randomness sources)
    )%%"))
      .def("SetExplicitUniformGeneratorSeed", &optimal_learning::RandomnessSourceContainer::SetExplicitUniformGeneratorSeed, R"%%(
    Seeds uniform generator with the specified seed value.

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("SetRandomizedUniformGeneratorSeed", &optimal_learning::RandomnessSourceContainer::SetRandomizedUniformGeneratorSeed, R"%%(
    Seeds uniform generator with info dependent on the current time.

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("ResetUniformRNGSeed", &optimal_learning::RandomnessSourceContainer::ResetUniformGeneratorState, R"%%(
    Resets Uniform RNG to most recently specified seed value.  Useful for testing
      )%%")
      .def("SetExplicitNormalRNGSeed", &optimal_learning::RandomnessSourceContainer::SetExplicitNormalRNGSeed, R"%%(
    Seeds RNG of thread i to f_i(seed, thread_id_i) such that f_i != f_j for i != j.  f_i is repeatable.
    So each thread gets a distinct seed that is easily repeatable for testing.

    NOTE: every thread is GUARANTEED to have a different seed

    INPUTS:
    seed: base seed value to use
      )%%")
      .def("SetRandomizedNormalRNGSeed", &optimal_learning::RandomnessSourceContainer::SetRandomizedNormalRNGSeed, R"%%(
    Set a new seed for the random number generator.  A "random" seed is selected based on
    the input seed value, the current time, and the thread_id.

    INPUTS:
    seed: base value for the new seed
    thread_id: id of the thread using this object
      )%%")
      .def("SetNormalRNGSeedPythonList", &optimal_learning::RandomnessSourceContainer::SetNormalRNGSeedPythonList, R"%%(
    If seed_flag_list[i] is true, sets the normal rng seed of the i-th thread to the value of seed_list[i].

    If sizes are invalid (i.e., number of seeds != number of generators), then no changes are made and an error code is returned.

    NOTE: Does not guarantee that all threads receive unique seeds!  If that is desired, seed_list should be
          checked BEFORE calling this function.

    RETURNS:
    0 if success, 1 if failure (due to invalid sizes)
      )%%")
      .def("ResetNormalRNGSeed", &optimal_learning::RandomnessSourceContainer::ResetNormalRNGState, R"%%(
    Resets all threads' RNGs to their most recently specified seed values.  Useful for testing
      )%%")
      .def("PrintState", &optimal_learning::RandomnessSourceContainer::PrintState, R"%%(
    Prints the state of the generator to stdout.  For testing.
      )%%")
      ;  // NOLINT, this is boost style
}  // end BOOST_PYTHON_MODULE(GPP) definition

}  // end unnamed namespace
