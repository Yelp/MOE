// gpp_python_expected_improvement.cpp
/*
  This file has the logic to invoke C++ functions pertaining to expected improvement from Python.
  The data flow follows the basic 4 step from gpp_python_common.hpp.

  Note: several internal functions of this source file are only called from Export*() functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  Export*() as Python docstrings, so we saw no need to repeat ourselves.
*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_expected_improvement.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <algorithm>  // NOLINT(build/include_order)
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include <boost/python/bases.hpp>  // NOLINT(build/include_order)
#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/def.hpp>  // NOLINT(build/include_order)
#include <boost/python/dict.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)
#include <boost/python/object.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_geometry.hpp"
#include "gpp_heuristic_expected_improvement_optimization.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {

double ComputeExpectedImprovementWrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample, int num_its, double best_so_far, bool force_monte_carlo, RandomnessSourceContainer& randomness_source) {
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

boost::python::list ComputeGradExpectedImprovementWrapper(const GaussianProcess& gaussian_process, const boost::python::list& points_to_sample, int num_to_sample, int num_its, double best_so_far, bool force_monte_carlo, RandomnessSourceContainer& randomness_source, const boost::python::list& current_point) {
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

/*
  Utility that dispatches EI optimization based on optimizer type and num_samples_to_generate.
  This is just used to reduce copy-pasted code.

  INPUTS:
  optimization_parameters: EPI/src/python/optimization_parameters.ExpectedImprovementOptimizationParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
  gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
    underlying GP
  input_container: PythonInterfaceInputContainer object containing data about points_to_sample
  domain: object specifying the domain to optimize over (see gpp_domain.hpp)
  domain_name: name of the domain, e.g., "tensor" or "simplex". Used to update the status dict
  optimizer_type: type of optimization to use (e.g., null, gradient descent)
  num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
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

boost::python::list MultistartExpectedImprovementOptimizationWrapper(const boost::python::object& optimization_parameters, const GaussianProcess& gaussian_process, const boost::python::list& domain, const boost::python::list& points_to_sample, int num_to_sample, int num_samples_to_generate, double best_so_far, int max_int_steps, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
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

/*
  Utility that dispatches heuristic EI optimization (solving q,0-EI) based on optimizer type and num_samples_to_generate.
  This is just used to reduce copy-pasted code.

  INPUTS:
  optimization_parameters: EPI/src/python/optimization_parameters.ExpectedImprovementOptimizationParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
  gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
    underlying GP
  domain: object specifying the domain to optimize over (see gpp_domain.hpp)
  domain_name: name of the domain, e.g., "tensor" or "simplex". Used to update the status dict
  optimizer_type: type of optimization to use (e.g., null, gradient descent)
  estimation_policy: the policy to use to produce (heuristic) objective function estimates during multi-points EI optimization
  num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
  best_so_far: value of the best sample so far (must be min(points_sampled_value))
  max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
  randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
  status: pydict object; cannot be None
  OUTPUTS:
  randomness_source: PRNG internal states modified
  status: modified on exit to describe whether convergence occurred
  best_points_to_sample[num_samples_to_generate][dim]: next set of points to evaluate
*/
template <typename DomainType>
void DispatchHeuristicExpectedImprovementOptimization(const boost::python::object& optimization_parameters, const GaussianProcess& gaussian_process, const DomainType& domain_object, const std::string& domain_name, OptimizerTypes optimizer_type, const ObjectiveEstimationPolicyInterface& estimation_policy, int num_samples_to_generate, double best_so_far, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status, double * restrict best_points_to_sample) {
  bool found_flag = false;
  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      // optimization_parameters must contain an int num_multistarts field, extract it
      int num_random_samples = boost::python::extract<int>(optimization_parameters.attr("num_random_samples"));

      bool random_search_only = true;
      GradientDescentParameters gradient_descent_parameters(0, 0, 0, 1.0, 1.0, 1.0, 0.0);  // dummy struct; we aren't using gradient descent
      ComputeHeuristicSetOfPointsToSample(gaussian_process, gradient_descent_parameters, domain_object, estimation_policy, best_so_far, max_num_threads, random_search_only, num_random_samples, num_samples_to_generate, &found_flag, &randomness_source.uniform_generator, best_points_to_sample);

      status["lhc_" + domain_name + "_domain_found_update"] = found_flag;
      break;
    }  // end case kNull optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimization_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimization_parameters.attr("optimizer_parameters"));
      int num_random_samples = boost::python::extract<int>(optimization_parameters.attr("num_random_samples"));

      bool random_search_only = false;
      ComputeHeuristicSetOfPointsToSample(gaussian_process, gradient_descent_parameters, domain_object, estimation_policy, best_so_far, max_num_threads, random_search_only, num_random_samples, num_samples_to_generate, &found_flag, &randomness_source.uniform_generator, best_points_to_sample);

      status["gradient_descent_" + domain_name + "_domain_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent optimizer_type
    default: {
      std::fill(best_points_to_sample, best_points_to_sample + gaussian_process.dim()*num_samples_to_generate, 0.0);
      OL_THROW_EXCEPTION(RuntimeException, "ERROR: invalid optimizer choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over optimizer_type
}

boost::python::list HeuristicExpectedImprovementOptimizationWrapper(const boost::python::object& optimization_parameters, const GaussianProcess& gaussian_process, const boost::python::list& domain, const ObjectiveEstimationPolicyInterface& estimation_policy, int num_samples_to_generate, double best_so_far, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
  // TODO(eliu): (#55793) make domain objects constructible from python; and pass them in through
  // the optimization_parameters python object
  int dim = gaussian_process.dim();
  std::vector<ClosedInterval> domain_C(dim);
  CopyPylistToClosedIntervalVector(domain, dim, domain_C);

  std::vector<double> best_points_to_sample_C(dim*num_samples_to_generate);

  DomainTypes domain_type = boost::python::extract<DomainTypes>(optimization_parameters.attr("domain_type"));
  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimization_parameters.attr("optimizer_type"));
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      TensorProductDomain domain_object(domain_C.data(), dim);
      std::string domain_name("tensor");

      DispatchHeuristicExpectedImprovementOptimization(optimization_parameters, gaussian_process, domain_object, domain_name, optimizer_type, estimation_policy, num_samples_to_generate, best_so_far, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kTensorProduct
    case DomainTypes::kSimplex: {
      SimplexIntersectTensorProductDomain domain_object(domain_C.data(), dim);
      std::string domain_name("simplex");

      DispatchHeuristicExpectedImprovementOptimization(optimization_parameters, gaussian_process, domain_object, domain_name, optimizer_type, estimation_policy, num_samples_to_generate, best_so_far, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
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

boost::python::list EvaluateEIAtPointListWrapper(const GaussianProcess& gaussian_process, const boost::python::list& initial_guesses, const boost::python::list& points_to_sample, int num_multistarts, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, RandomnessSourceContainer& randomness_source, boost::python::dict& status) {
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

}  // end unnamed namespace

void ExportEstimationPolicies() {
  boost::python::class_<ObjectiveEstimationPolicyInterface, boost::noncopyable>("ObjectiveEstimationPolicyInterface", R"%%(
    Pure abstract (in the C++ sense) base class for objective function estimation (e.g., Constant Liar, Kriging Believer).
    Serves no purpose in Python but is needed by boost to allow pointer casts of derived types to this type.
    )%%", boost::python::no_init);

  boost::python::class_<ConstantLiarEstimationPolicy, boost::python::bases<ObjectiveEstimationPolicyInterface> >("ConstantLiarEstimationPolicy", R"%%(
    Produces objective function estimates at a "point" using the "Constant Liar" heuristic.

    Always outputs function_value = lie_value (the "constant lie") and noise_variance = lie_noise_variance, regardless
    of the value of "point".

    members:
    double lie_value: the "constant lie" that this estimator should return
    double lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
    )%%", boost::python::init<double, double>(R"%%(
    Constructs a ConstantLiarEstimationPolicy object.

    double lie_value: the "constant lie" that this estimator should return
    double lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
    )%%"));

  boost::python::class_<KrigingBelieverEstimationPolicy, boost::python::bases<ObjectiveEstimationPolicyInterface> >("KrigingBelieverEstimationPolicy", R"%%(
    Produces objective function estimates at a "point" using the "Kriging Believer" heuristic.

    Requires a valid GaussianProcess (GP) to produce estimates. Computes estimates as:
    function_value = GP.Mean(point) + std_deviation_coef * sqrt(GP.Variance(point))
    noise_variance = kriging_noise_variance

    members:
    double std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    double kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    )%%", boost::python::init<double, double>(R"%%(
    Constructs for KrigingBelieverEstimationPolicy object.

    double std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    double kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    )%%"));
}

void ExportExpectedImprovementFunctions() {
  boost::python::def("compute_expected_improvement", ComputeExpectedImprovementWrapper, R"%%(
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

  boost::python::def("compute_grad_expected_improvement", ComputeGradExpectedImprovementWrapper, R"%%(
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

  boost::python::def("multistart_expected_improvement_optimization", MultistartExpectedImprovementOptimizationWrapper, R"%%(
    Optimize expected improvement (i.e., solve q,p-EI) over the specified domain using the specified optimization method.
    Can optimize for num_samples_to_generate new points to sample (i.e., aka "q", experiments to run) simultaneously.
    Allows the user to specify num_to_sample (aka "p") ongoing/concurrent experiments.

    The ExpectedImprovementOptimizationParameters object is a python class defined in:
    EPI/src/python/optimization_parameters.ExpectedImprovementOptimizationParameters
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
    int num_to_sample: number of points to sample (i.e., the p in q,p-EI)
    int num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    double best_so_far: best known value of objective so far
    int max_int_steps: number of MC integration points in EI and grad_EI
    int max_num_threads: max number of threads to use during EI optimization
    RandomnessSourceContainer randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    RETURNS:
    pylist result[num_samples_to_generate][dim]: next set of points to eval
    )%%");

  boost::python::def("heuristic_expected_improvement_optimization", HeuristicExpectedImprovementOptimizationWrapper, R"%%(
    Compute a heuristic approximation to the result of multistart_expected_improvement_optimization(). That is, it
    optimizes an approximation to q,0-EI over the specified domain using the specified optimization method.
    Can optimize for num_samples_to_generate (aka "q") new points to sample (i.e., experiments to run) simultaneously.

    Computing q,p-EI for q > 1 or p > 1 is expensive. To avoid that cost, this method "solves" q,0-EI by repeatedly
    optimizing 1,0-EI. We do the following (in C++):
    for i in range(num_samples_to_generate)
      new_point = optimize_1_EI(gaussian_process, ...)
      new_function_value, new_noise_variance = estimation_policy.compute_estimate(new_point, gaussian_process, i)
      gaussian_process.add_point(new_point, new_function_value, new_noise_variance)
    So using estimation_policy, we guess what the real-world objective function value would be, evaluated at the result
    of each 1-EI optimization. Then we treat this estimate as *truth* and feed it back to the gaussian process. The
    ConstantLiar and KrigingBelieverEstimationPolicy objects reproduce the heuristics described in Ginsbourger 2008.

    See gpp_heuristic_expected_improvement_optimization.hpp for further details on the algorithm.

    The ExpectedImprovementOptimizationParameters object is a python class defined in:
    EPI/src/python/optimization_parameters.ExpectedImprovementOptimizationParameters
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
    ObjectiveEstimationPolicyInterface estimation_policy: the policy to use to produce (heuristic) objective function estimates
      during q,0-EI optimization (e.g., ConstantLiar, KrigingBeliever)
    int num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    double best_so_far: best known value of objective so far
    int max_num_threads: max number of threads to use during EI optimization
    RandomnessSourceContainer randomness_source: object containing at least a UniformRandomGenerator randomness source
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    RETURNS:
    pylist result[num_samples_to_generate][dim]: next set of points to eval
    )%%");

  boost::python::def("evaluate_EI_at_point_list", EvaluateEIAtPointListWrapper, R"%%(
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
    int num_to_sample: number of points to sample (i.e., the p in 1,p-EI)
    double best_so_far: best known value of objective so far
    int max_int_steps: number of MC integration points in EI and grad_EI
    int max_num_threads: max number of threads to use during EI optimization
    RandomnessSourceContainer randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    pydict status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred

    OUTPUTS:
    pylist result[num_multistarts]: EI values at each point of the initial_guesses list, in the same order
    )%%");
}

}  // end namespace optimal_learning
