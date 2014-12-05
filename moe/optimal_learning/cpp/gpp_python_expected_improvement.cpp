/*!
  \file gpp_python_expected_improvement.cpp
  \rst
  This file has the logic to invoke C++ functions pertaining to expected improvement from Python.
  The data flow follows the basic 4 step from gpp_python_common.hpp.

  .. NoteL: several internal functions of this source file are only called from ``Export*()`` functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  ``Export*()`` as Python docstrings, so we saw no need to repeat ourselves.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_expected_improvement.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
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
#include "gpp_exception.hpp"
#include "gpp_expected_improvement_gpu.hpp"
#include "gpp_geometry.hpp"
#include "gpp_heuristic_expected_improvement_optimization.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {

double ComputeExpectedImprovementWrapper(const GaussianProcess& gaussian_process,
                                         const boost::python::list& points_to_sample,
                                         const boost::python::list& points_being_sampled,
                                         int num_to_sample, int num_being_sampled,
                                         int max_int_steps, double best_so_far,
                                         bool force_monte_carlo,
                                         RandomnessSourceContainer& randomness_source) {
  PythonInterfaceInputContainer input_container(points_to_sample, points_being_sampled,
                                                gaussian_process.dim(), num_to_sample, num_being_sampled);

  bool configure_for_gradients = false;
  if ((num_to_sample == 1) && (num_being_sampled == 0) && (force_monte_carlo == false)) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator,
                                                                       input_container.points_to_sample.data(),
                                                                       configure_for_gradients);
    return ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, input_container.points_to_sample.data(),
                                                     input_container.points_being_sampled.data(),
                                                     input_container.num_to_sample,
                                                     input_container.num_being_sampled,
                                                     configure_for_gradients,
                                                     randomness_source.normal_rng_vec.data());
    return ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }
}

boost::python::list ComputeGradExpectedImprovementWrapper(const GaussianProcess& gaussian_process,
                                                          const boost::python::list& points_to_sample,
                                                          const boost::python::list& points_being_sampled,
                                                          int num_to_sample, int num_being_sampled,
                                                          int max_int_steps, double best_so_far,
                                                          bool force_monte_carlo,
                                                          RandomnessSourceContainer& randomness_source) {
  PythonInterfaceInputContainer input_container(points_to_sample, points_being_sampled, gaussian_process.dim(),
                                                num_to_sample, num_being_sampled);

  std::vector<double> grad_EI(num_to_sample*input_container.dim);
  bool configure_for_gradients = true;
  if ((num_to_sample == 1) && (num_being_sampled == 0) && (force_monte_carlo == false)) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator,
                                                                       input_container.points_to_sample.data(),
                                                                       configure_for_gradients);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_EI.data());
  } else {
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, input_container.points_to_sample.data(),
                                                     input_container.points_being_sampled.data(),
                                                     input_container.num_to_sample,
                                                     input_container.num_being_sampled,
                                                     configure_for_gradients,
                                                     randomness_source.normal_rng_vec.data());
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_EI.data());
  }

  return VectorToPylist(grad_EI);
}

/*!\rst
  Utility that dispatches EI optimization based on optimizer type and num_to_sample.
  This is just used to reduce copy-pasted code.

  \param
    :optimizer_parameters: python/cpp_wrappers/optimization._CppOptimizerParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
    :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
      underlying GP
    :input_container: PythonInterfaceInputContainer object containing data about points_being_sampled
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :optimizer_type: type of optimization to use (e.g., null, gradient descent)
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :best_so_far: value of the best sample so far (must be min(points_sampled_value))
    :max_int_steps: maximum number of MC iterations
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    :status: pydict object; cannot be None
  \output
    :randomness_source: PRNG internal states modified
    :status: modified on exit to describe whether convergence occurred
    :best_points_to_sample[num_to_sample][dim]: next set of points to evaluate
\endrst*/
template <typename DomainType>
void DispatchExpectedImprovementOptimization(const boost::python::object& optimizer_parameters,
                                             const GaussianProcess& gaussian_process,
                                             const PythonInterfaceInputContainer& input_container,
                                             const DomainType& domain,
                                             OptimizerTypes optimizer_type,
                                             int num_to_sample, double best_so_far,
                                             int max_int_steps, int max_num_threads,
                                             bool use_gpu, int which_gpu,
                                             RandomnessSourceContainer& randomness_source,
                                             boost::python::dict& status,
                                             double * restrict best_points_to_sample) {
#ifndef OL_GPU_ENABLED
  (void) which_gpu;  // quiet the compiler warning (unused variable)
#endif

  bool found_flag = false;
  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      ThreadSchedule thread_schedule(max_num_threads, omp_sched_static);
      // optimizer_parameters must contain an int num_random_samples field, extract it
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      if (use_gpu == true) {
#ifdef OL_GPU_ENABLED
        CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, domain, thread_schedule,
                                                                input_container.points_being_sampled.data(),
                                                                num_random_samples, num_to_sample,
                                                                input_container.num_being_sampled,
                                                                best_so_far, max_int_steps, which_gpu, &found_flag,
                                                                &randomness_source.uniform_generator,
                                                                best_points_to_sample);
#else
        OL_THROW_EXCEPTION(OptimalLearningException, "GPU is not installed or enabled!");
#endif
      } else {
        ComputeOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, domain, thread_schedule,
                                                            input_container.points_being_sampled.data(),
                                                            num_random_samples, num_to_sample,
                                                            input_container.num_being_sampled,
                                                            best_so_far, max_int_steps,
                                                            &found_flag, &randomness_source.uniform_generator,
                                                            randomness_source.normal_rng_vec.data(),
                                                            best_points_to_sample);
      }
      status[std::string("lhc_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kNull optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimizer_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));
      ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      bool random_search_only = false;
      if (use_gpu == true) {
#ifdef OL_GPU_ENABLED
        CudaComputeOptimalPointsToSample(gaussian_process, gradient_descent_parameters, domain, thread_schedule,
                                         input_container.points_being_sampled.data(), num_to_sample,
                                         input_container.num_being_sampled, best_so_far, max_int_steps,
                                         random_search_only, num_random_samples, which_gpu, &found_flag,
                                         &randomness_source.uniform_generator, best_points_to_sample);
#else
        OL_THROW_EXCEPTION(OptimalLearningException, "GPU is not installed or enabled!");
#endif
      } else {
        ComputeOptimalPointsToSample(gaussian_process, gradient_descent_parameters, domain, thread_schedule,
                                     input_container.points_being_sampled.data(), num_to_sample,
                                     input_container.num_being_sampled, best_so_far, max_int_steps,
                                     random_search_only, num_random_samples, &found_flag,
                                     &randomness_source.uniform_generator,
                                     randomness_source.normal_rng_vec.data(), best_points_to_sample);
      }
      status[std::string("gradient_descent_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent optimizer_type
    default: {
      std::fill(best_points_to_sample, best_points_to_sample + input_container.dim*num_to_sample, 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid optimizer choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over optimizer_type
}

boost::python::list MultistartExpectedImprovementOptimizationWrapper(const boost::python::object& optimizer_parameters,
                                                                     const GaussianProcess& gaussian_process,
                                                                     const boost::python::list& domain_bounds,
                                                                     const boost::python::list& points_being_sampled,
                                                                     int num_to_sample, int num_being_sampled,
                                                                     double best_so_far, int max_int_steps,
                                                                     int max_num_threads, bool use_gpu, int which_gpu,
                                                                     RandomnessSourceContainer& randomness_source,
                                                                     boost::python::dict& status) {
  // TODO(GH-131): make domain objects constructible from python; and pass them in through
  // the optimizer_parameters python object

  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  int num_to_sample_input = 0;  // No points to sample; we are generating these via EI optimization
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(points_to_sample_dummy, points_being_sampled, gaussian_process.dim(), num_to_sample_input, num_being_sampled);
  std::vector<ClosedInterval> domain_bounds_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain_bounds, input_container.dim, domain_bounds_C);

  std::vector<double> best_points_to_sample_C(input_container.dim*num_to_sample);

  DomainTypes domain_type = boost::python::extract<DomainTypes>(optimizer_parameters.attr("domain_type"));
  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimizer_parameters.attr("optimizer_type"));
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      TensorProductDomain domain(domain_bounds_C.data(), input_container.dim);

      DispatchExpectedImprovementOptimization(optimizer_parameters, gaussian_process, input_container,
                                              domain, optimizer_type, num_to_sample, best_so_far,
                                              max_int_steps, max_num_threads, use_gpu, which_gpu,
                                              randomness_source,
                                              status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kTensorProduct
    case DomainTypes::kSimplex: {
      SimplexIntersectTensorProductDomain domain(domain_bounds_C.data(), input_container.dim);

      DispatchExpectedImprovementOptimization(optimizer_parameters, gaussian_process, input_container,
                                              domain, optimizer_type, num_to_sample, best_so_far,
                                              max_int_steps, max_num_threads, use_gpu, which_gpu,
                                              randomness_source,
                                              status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kSimplex
    default: {
      std::fill(best_points_to_sample_C.begin(), best_points_to_sample_C.end(), 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid domain choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over domain_type

  return VectorToPylist(best_points_to_sample_C);
}

/*!\rst
  Utility that dispatches heuristic EI optimization (solving q,0-EI) based on optimizer type and num_to_sample.
  This is just used to reduce copy-pasted code.

  \param
    :optimizer_parameters: python/cpp_wrappers/optimization._CppOptimizerParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
    :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
      underlying GP
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :optimizer_type: type of optimization to use (e.g., null, gradient descent)
    :estimation_policy: the policy to use to produce (heuristic) objective function estimates during multi-points EI optimization
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :best_so_far: value of the best sample so far (must be min(points_sampled_value))
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :randomness_source: object containing randomness sources (sufficient for multithreading) used in EI computation
    :status: pydict object; cannot be None
  \output
    :randomness_source: PRNG internal states modified
    :status: modified on exit to describe whether convergence occurred
    :best_points_to_sample[num_to_sample][dim]: next set of points to evaluate
\endrst*/
template <typename DomainType>
void DispatchHeuristicExpectedImprovementOptimization(const boost::python::object& optimizer_parameters,
                                                      const GaussianProcess& gaussian_process,
                                                      const DomainType& domain,
                                                      OptimizerTypes optimizer_type,
                                                      const ObjectiveEstimationPolicyInterface& estimation_policy,
                                                      int num_to_sample, double best_so_far, int max_num_threads,
                                                      RandomnessSourceContainer& randomness_source,
                                                      boost::python::dict& status,
                                                      double * restrict best_points_to_sample) {
  ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);
  bool found_flag = false;
  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      // optimizer_parameters must contain an int num_multistarts field, extract it
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      bool random_search_only = true;
      GradientDescentParameters gradient_descent_parameters(0, 0, 0, 0, 1.0, 1.0, 1.0, 0.0);  // dummy struct; we aren't using gradient descent
      ComputeHeuristicPointsToSample(gaussian_process, gradient_descent_parameters, domain,
                                     estimation_policy, thread_schedule, best_so_far,
                                     random_search_only, num_random_samples, num_to_sample,
                                     &found_flag, &randomness_source.uniform_generator,
                                     best_points_to_sample);

      status[std::string("lhc_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kNull optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimizer_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      bool random_search_only = false;
      ComputeHeuristicPointsToSample(gaussian_process, gradient_descent_parameters, domain,
                                     estimation_policy, thread_schedule, best_so_far,
                                     random_search_only, num_random_samples, num_to_sample,
                                     &found_flag, &randomness_source.uniform_generator,
                                     best_points_to_sample);

      status[std::string("gradient_descent_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent optimizer_type
    default: {
      std::fill(best_points_to_sample, best_points_to_sample + gaussian_process.dim()*num_to_sample, 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid optimizer choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over optimizer_type
}

boost::python::list HeuristicExpectedImprovementOptimizationWrapper(const boost::python::object& optimizer_parameters,
                                                                    const GaussianProcess& gaussian_process,
                                                                    const boost::python::list& domain_bounds,
                                                                    const ObjectiveEstimationPolicyInterface& estimation_policy,
                                                                    int num_to_sample, double best_so_far, int max_num_threads,
                                                                    RandomnessSourceContainer& randomness_source,
                                                                    boost::python::dict& status) {
  // TODO(GH-131): make domain objects constructible from python; and pass them in through
  // the optimizer_parameters python object
  int dim = gaussian_process.dim();
  std::vector<ClosedInterval> domain_bounds_C(dim);
  CopyPylistToClosedIntervalVector(domain_bounds, dim, domain_bounds_C);

  std::vector<double> best_points_to_sample_C(dim*num_to_sample);

  DomainTypes domain_type = boost::python::extract<DomainTypes>(optimizer_parameters.attr("domain_type"));
  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimizer_parameters.attr("optimizer_type"));
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      TensorProductDomain domain(domain_bounds_C.data(), dim);

      DispatchHeuristicExpectedImprovementOptimization(optimizer_parameters, gaussian_process, domain,
                                                       optimizer_type, estimation_policy, num_to_sample,
                                                       best_so_far, max_num_threads, randomness_source,
                                                       status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kTensorProduct
    case DomainTypes::kSimplex: {
      SimplexIntersectTensorProductDomain domain(domain_bounds_C.data(), dim);

      DispatchHeuristicExpectedImprovementOptimization(optimizer_parameters, gaussian_process, domain,
                                                       optimizer_type, estimation_policy, num_to_sample,
                                                       best_so_far, max_num_threads, randomness_source,
                                                       status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kSimplex
    default: {
      std::fill(best_points_to_sample_C.begin(), best_points_to_sample_C.end(), 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid domain choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over domain_type

  return VectorToPylist(best_points_to_sample_C);
}

boost::python::list EvaluateEIAtPointListWrapper(const GaussianProcess& gaussian_process,
                                                 const boost::python::list& initial_guesses,
                                                 const boost::python::list& points_being_sampled,
                                                 int num_multistarts, int num_to_sample,
                                                 int num_being_sampled, double best_so_far,
                                                 int max_int_steps, int max_num_threads,
                                                 RandomnessSourceContainer& randomness_source,
                                                 boost::python::dict& status) {
  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  int num_to_sample_input = 0;  // No points to sample; we are generating these via EI optimization
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(points_to_sample_dummy, points_being_sampled, gaussian_process.dim(),
                                                num_to_sample_input, num_being_sampled);
  std::vector<double> result_point_C(input_container.dim);  // not used
  std::vector<double> result_function_values_C(num_multistarts);
  std::vector<double> initial_guesses_C(input_container.dim * num_multistarts);

  CopyPylistToVector(initial_guesses, input_container.dim * num_multistarts, initial_guesses_C);

  ThreadSchedule thread_schedule(max_num_threads, omp_sched_static);
  bool found_flag = false;
  EvaluateEIAtPointList(gaussian_process, thread_schedule, initial_guesses_C.data(),
                        input_container.points_being_sampled.data(), num_multistarts,
                        num_to_sample, input_container.num_being_sampled, best_so_far,
                        max_int_steps, &found_flag, randomness_source.normal_rng_vec.data(),
                        result_function_values_C.data(), result_point_C.data());

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

    :ivar lie_value: (*float64*) the "constant lie" that this estimator should return
    :ivar lie_noise_variance: (*float64*) the noise_variance to associate to the lie_value (MUST be >= 0.0)
    )%%", boost::python::init<double, double>(R"%%(
    Constructs a ConstantLiarEstimationPolicy object.

    :param lie_value: the "constant lie" that this estimator should return
    :type lie_value: float64 (finite)
    :param lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
    :type lie_noise_variance: float64 >= 0.0
    )%%"));

  boost::python::class_<KrigingBelieverEstimationPolicy, boost::python::bases<ObjectiveEstimationPolicyInterface> >("KrigingBelieverEstimationPolicy", R"%%(
    Produces objective function estimates at a "point" using the "Kriging Believer" heuristic.

    Requires a valid GaussianProcess (GP) to produce estimates. Computes estimates as:

    * function_value = GP.Mean(point) + std_deviation_coef * sqrt(GP.Variance(point))
    * noise_variance = kriging_noise_variance

    :ivar std_deviation_coef: (*float64*) the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    :ivar kriging_noise_variance: (*float64*) the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    )%%", boost::python::init<double, double>(R"%%(
    Constructs for KrigingBelieverEstimationPolicy object.

    :param std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    :type std_deviation_coef: float64 (finite)
    :param kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    :type kriging_noise_variance: float64 >= 0.0
    )%%"));
}

void ExportExpectedImprovementFunctions() {
  boost::python::def("compute_expected_improvement", ComputeExpectedImprovementWrapper, R"%%(
    Compute expected improvement.
    If ``num_to_sample == 1`` and ``num_being_sampled == 0`` AND ``force_monte_carlo is false``, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based EI computation is used.

    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param points_to_sample: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :type points_to_sample: list of float64 with shape (num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param force_monte_carlo: true to force monte carlo evaluation of EI
    :type force_monte_carlo: bool
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :return: computed EI
    :rtype: float64 >= 0.0
    )%%");

  boost::python::def("compute_grad_expected_improvement", ComputeGradExpectedImprovementWrapper, R"%%(
    Compute the gradient of expected improvement evaluated at points_to_sample.
    If num_to_sample = 1 and num_being_sampled = 0 AND force_monte_carlo is false, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based EI computation is used.

    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param points_to_sample: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :type points_to_sample: list of float64 with shape (num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param force_monte_carlo: true to force monte carlo evaluation of EI
    :type force_monte_carlo: bool
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :return: gradient of EI (computed at points_to_sample + points_being_sampled, wrt points_to_sample)
    :rtype: list of float64 with shape (num_to_sample, dim)
    )%%");

  boost::python::def("multistart_expected_improvement_optimization", MultistartExpectedImprovementOptimizationWrapper, R"%%(
    Optimize expected improvement (i.e., solve q,p-EI) over the specified domain using the specified optimization method.
    Can optimize for num_to_sample new points to sample (i.e., aka "q", experiments to run) simultaneously.
    Allows the user to specify num_being_sampled (aka "p") ongoing/concurrent experiments.

    The _CppOptimizerParameters object is a python class defined in:
    python/cpp_wrappers/optimization._CppOptimizerParameters
    See that class definition for more details.

    This function expects it to have the fields:

    * domain_type (DomainTypes enum from this file)
    * optimizer_type (OptimizerTypes enum from this file)
    * num_random_samples (int, number of samples to 'dumb' search over, if 'dumb' search is being used.
      e.g., if optimizer = kNull or if to_sample > 1)
    * optimizer_parameters (*Parameters struct (gpp_optimizer_parameters.hpp) where * matches optimizer_type
      unused if optimizer_type == kNull)

    This function also has the option of using GPU to compute general q,p-EI via MC simulation. To enable it,
    make sure you have installed GPU components of MOE, otherwise, it will throw Runtime excpetion.

    .. WARNING:: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads

    :param optimizer_parameters: python object containing the DomainTypes domain_type and
      OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton)
    :type optimizer_parameters: _CppOptimizerParameters
    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param domain: [lower, upper] bound pairs for each dimension
    :type domain: list of float64 with shape (dim, 2)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param max_num_threads: max number of threads to use during EI optimization
    :type max_num_threads: int >= 1
    :param use_gpu: set to 1 if user wants to use GPU for MC computation
    :type use_gpu: bool
    :param which_gpu: GPU device ID
    :type which_gpu: int >= 0
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :param status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred
    :type status: dict
    :return: next set of points to eval
    :rtype: list of float64 with shape (num_to_sample, dim)
    )%%");

  boost::python::def("heuristic_expected_improvement_optimization", HeuristicExpectedImprovementOptimizationWrapper, R"%%(
    Compute a heuristic approximation to the result of multistart_expected_improvement_optimization(). That is, it
    optimizes an approximation to q,0-EI over the specified domain using the specified optimization method.
    Can optimize for num_to_sample (aka "q") new points to sample (i.e., experiments to run) simultaneously.

    Computing q,p-EI for q > 1 or p > 1 is expensive. To avoid that cost, this method "solves" q,0-EI by repeatedly
    optimizing 1,0-EI. We do the following (in C++)::

      for i in xrange(num_to_sample):
        new_point = optimize_1_EI(gaussian_process, ...)
        new_function_value, new_noise_variance = estimation_policy.compute_estimate(new_point, gaussian_process, i)
        gaussian_process.add_point(new_point, new_function_value, new_noise_variance)

    So using estimation_policy, we guess what the real-world objective function value would be, evaluated at the result
    of each 1-EI optimization. Then we treat this estimate as *truth* and feed it back to the gaussian process. The
    ConstantLiar and KrigingBelieverEstimationPolicy objects reproduce the heuristics described in Ginsbourger 2008.

    See gpp_heuristic_expected_improvement_optimization.hpp for further details on the algorithm.

    The _CppOptimizerParameters object is a python class defined in:
    ``python/cpp_wrappers/optimization._CppOptimizerParameters``
    See that class definition for more details.

    This function expects it to have the fields:

    * domain_type (DomainTypes enum from this file)
    * optimizer_type (OptimizerTypes enum from this file)
    * num_random_samples (int, number of samples to 'dumb' search over, if 'dumb' search is being used.
      e.g., if optimizer = kNull or if to_sample > 1)
    * optimizer_parameters (*Parameters struct (gpp_optimizer_parameters.hpp) where * matches optimizer_type
      unused if optimizer_type == kNull)

    .. WARNING:: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads

    :param optimizer_parameters: python object containing the DomainTypes domain_type and
      OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton)
    :type optimizer_parameters: _CppOptimizerParameters
    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param domain: [lower, upper] bound pairs for each dimension
    :type domain: list of float64 with shape (dim, 2)
    :param estimation_policy: the policy to use to produce (heuristic) objective function estimates
      during q,0-EI optimization (e.g., ConstantLiar, KrigingBeliever)
    :type estimation_policy: ObjectiveEstimationPolicyInterface
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param max_num_threads: max number of threads to use during EI optimization
    :type max_num_threads: int >= 1
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :param status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred
    :type status: dict
    :return: next set of points to eval
    :rtype: list of float64 with shape (num_to_sample, dim)
    )%%");

  boost::python::def("evaluate_EI_at_point_list", EvaluateEIAtPointListWrapper, R"%%(
    Evaluates the expected improvement at each point in initial_guesses; can handle q,p-EI.
    Useful for plotting.

    Equivalent to::

      result = []
      for point in initial_guesses:
          result.append(compute_expected_improvement(point, ...))

    But this method is substantially faster (loop in C++ and multithreaded).

    .. WARNING:: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads


    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0


    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param initial_guesses: points at which to evaluate EI
    :type initial_guesses: list of flaot64 with shape (num_multistarts, num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_multistarts: number of points at which to evaluate EI
    :type num_multistarts: int > 0
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param max_num_threads: max number of threads to use during EI optimization
    :type max_num_threads: int >= 1
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :param status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred
    :type status: dict
    :return: EI values at each point of the initial_guesses list, in the same order
    :rtype: list of float64 with shape (num_multistarts, )
    )%%");
}

}  // end namespace optimal_learning
