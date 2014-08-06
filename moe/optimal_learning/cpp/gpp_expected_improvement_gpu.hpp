/*!
  \file gpp_expected_improvement_gpu.hpp
  \rst
  All GPU related functions are declared here, and any other C++ functions who wish to call GPU functions should only call functions here.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_

#include <algorithm>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"

#ifdef OL_GPU_ENABLED

#include "gpu/gpp_cuda_math.hpp"
/*!\rst
  Macro that checks error message (CudaError object) returned by CUDA functions, and throws
  OptimalLearningCudaException if there is error.
\endrst*/
#define OL_CUDA_ERROR_THROW(X) do {CudaError _ERR = (X); if ((_ERR).err != cudaSuccess) {ThrowException(OptimalLearningCudaException(_ERR));}} while (0)

#endif

namespace optimal_learning {
#ifdef OL_GPU_ENABLED

/*!\rst
  This struct does the same job as C++ smart pointer. It contains pointer to memory location on
  GPU, its constructor and destructor also take care of memory allocation/deallocation on GPU. 
\endrst*/
struct CudaDevicePointer final {
  explicit CudaDevicePointer(int num_doubles_in);

  ~CudaDevicePointer();

  //! pointer to the memory location on gpu
  double* ptr;
  //! number of doubles to allocate on gpu, so the memory size is num_doubles * sizeof(double)
  int num_doubles;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaDevicePointer);
};

/*!\rst
  Exception to handle runtime errors returned by CUDA API functions. This class subclasses
  OptimalLearningException in gpp_exception.hpp/cpp, and basiclly has the same functionality
  as its superclass, except the constructor is different.
\endrst*/
class OptimalLearningCudaException : public OptimalLearningException {
 public:
  //! String name of this exception ofr logging.
  constexpr static char const * kName = "OptimalLearningCudaException";

  /*!\rst
    Constructs a OptimalLearningCudaException with struct CudaError
    \param
      :error: C struct that contains error message returned by CUDA API functions
  \endrst*/
  explicit OptimalLearningCudaException(const CudaError& error);

  OL_DISALLOW_DEFAULT_AND_ASSIGN(OptimalLearningCudaException);
};

struct CudaExpectedImprovementState;

/*!\rst
  This class has the same functionality as ExpectedImprovementEvaluator (see gpp_math.hpp),
  except that computations are performed on GPU.
\endrst*/
class CudaExpectedImprovementEvaluator final {
 public:
  using StateType = CudaExpectedImprovementState;
  /*!\rst
    Constructor that also specify which gpu you want to use (for multi-gpu system)
  \endrst*/
  CudaExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in,
                                   int num_mc_in, double best_so_far, int devID_in);

  ~CudaExpectedImprovementEvaluator();

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_mc() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_mc_;
  }

  const GaussianProcess * gaussian_process() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return gaussian_process_;
  }

  /*!\rst
    Wrapper for ComputeExpectedImprovement(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeExpectedImprovement(ei_state);
  }

  /*!\rst
    Wrapper for ComputeGradExpectedImprovement(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * ei_state, double * restrict grad_ei) const OL_NONNULL_POINTERS {
    ComputeGradExpectedImprovement(ei_state, grad_ei);
  }

  /*!\rst
    This function has the same functionality as ComputeExpectedImprovement (see gpp_math.hpp)
    in class ExpectedImprovementEvaluator.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``uniform_rng`` modified
    \return
      the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    This function has the same functionality as ComputeGradExpectedImprovement (see gpp_math.hpp)
    in class ExpectedImprovementEvaluator.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``uniform_rng`` modified
      :grad_ei[dim][num_to_sample]: gradient of EI
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_ei) const OL_NONNULL_POINTERS;

  /*!\rst
    Call CUDA API function to activate a GPU.
    Refer to: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g418c299b069c4803bfb7cab4943da383

    \param
      :devID: device ID of the GPU need to be activated
  \endrst*/
  void SetupGPU(int devID);

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! number of mc iterations
  int num_mc_;
  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;
  //! pointer to gaussian process used in EI computations
  const GaussianProcess * gaussian_process_;
};

/*!\rst
  This has the same functionality as ExpectedImprovementState (see gpp_math.hpp) except that it is for GPU computing
\endrst*/
struct CudaExpectedImprovementState final {
  using EvaluatorType = CudaExpectedImprovementEvaluator;

  /*!\rst
    Constructs an CudaExpectedImprovementState object with a specified source of randomness for
    the purpose of computing EI(and its gradient) over the specified set of points to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent
    state from the associated Gaussian Process (which arrives as part of the ei_evaluator).

    .. WARNING:: This object is invalidated if the associated ei_evaluator is mutated.  SetupState()
    should be called to reset.

    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in
         UNDEFINED BEHAVIOR.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :points_to_sample[dim][num_to_sample]: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :points_being_sampled[dim][num_being_sampled]: points being sampled in concurrent experiments
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
      :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
      :uniform_rng[1]: pointer to a properly initialized* UniformRandomGenerator object

    .. NOTE::
         * The UniformRandomGenerator object must already be seeded.  If multithreaded computation is used for EI, then every state object
         must have a different UniformRandomGenerator (different seeds, not just different objects).
  \endrst*/
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                               double const * restrict points_being_sampled, int num_to_sample_in,
                               int num_being_sampled_in, bool configure_for_gradients,
                               UniformRandomGenerator* uniform_rng_in);

  // constructor for setting up unit test
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                               double const * restrict points_being_sampled, int num_to_sample_in,
                               int num_being_sampled_in, bool configure_for_gradients,
                               UniformRandomGenerator * uniform_rng_in, bool configure_for_test);

  CudaExpectedImprovementState(CudaExpectedImprovementState&& OL_UNUSED(other)) = default;

  /*!\rst
    Create a vector with the union of points_to_sample and points_being_sampled (the latter is appended to the former).

    Note the l-value return. Assigning the return to a std::vector<double> or passing it as an argument to the ctor
    will result in copy-elision or move semantics; no copying/performance loss.

    \param:
      :points_to_sample[dim][num_to_sample]: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :points_being_sampled[dim][num_being_sampled]: points being sampled in concurrent experiments
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
      :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
      :dim: the number of spatial dimensions of each point array
    \return
      std::vector<double> with the union of the input arrays: points_being_sampled is *appended* to points_to_sample
  \endrst*/
  static std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample,
                                                double const * restrict points_being_sampled,
                                                int num_to_sample, int num_being_sampled, int dim)
                                                noexcept OL_WARN_UNUSED_RESULT;

  /*!\rst
    A simple utility function to calculate how many random numbers will be generated by GPU computation of EI/gradEI given
    number of MC simulations. (user set num_mc_itr is not necessarily equal to the actual num_mc_itr used in GPU computation,
    because actual num_mc_itr has to be multiple of (num_threads * num_blocks)

    \param:
      :num_mc_itr: number of MC simulations
      :num_threads: number of threads per block in GPU computation
      :num_blocks: number of blocks in GPU computation
      :num_points: number of points interested (aka q+p)
    \return
      int: number of random numbers generated in GPU computation
  \endrst*/
  static int GetVectorSize(int num_mc_itr, int num_threads, int num_blocks, int num_points) noexcept OL_WARN_UNUSED_RESULT;

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim*num_to_sample;
  }

  /*!\rst
    Get the ``points_to_sample``: potential future samples whose EI (and/or gradients) are being evaluated

    \output
      :points_to_sample[dim][num_to_sample]: potential future samples whose EI (and/or gradients) are being evaluated
  \endrst*/
  void GetCurrentPoint(double * restrict points_to_sample) const noexcept OL_NONNULL_POINTERS {
    std::copy(union_of_points.data(), union_of_points.data() + num_to_sample*dim, points_to_sample);
  }

  /*!\rst
    Change the potential samples whose EI (and/or gradient) are being evaluated.
    Update the state's derived quantities to be consistent with the new points.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :points_to_sample[dim][num_to_sample]: potential future samples whose EI (and/or gradients) are being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with new ``points_to_sample``, the location of the potential samples whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.

    .. WARNING::
         This object's state is INVALIDATED if the ``ei_evaluator`` (including the GaussianProcess it depends on) used in
         SetupState is mutated! SetupState() should be called again in such a situation.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :points_to_sample[dim][num_to_sample]: potential future samples whose EI (and/or gradients) are being evaluated
  \endrst*/
  void SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  const int num_to_sample;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  const int num_being_sampled;
  //! number of derivative terms desired (usually 0 for no derivatives or num_to_sample)
  const int num_derivatives;
  //! number of points in union_of_points: num_to_sample + num_being_sampled
  const int num_union;

  //! points currently being sampled; this is the union of the points represented by "q" and "p" in q,p-EI
  //! ``points_to_sample`` is stored first in memory, immediately followed by ``points_being_sampled``
  std::vector<double> union_of_points;

  //! gaussian process state
  GaussianProcess::StateType points_to_sample_state;

  //! random number generator
  UniformRandomGenerator* uniform_rng;

  // temporary storage: preallocated space used by CudaExpectedImprovementEvaluator's member functions
  //! the mean of the GP evaluated at union_of_points
  std::vector<double> to_sample_mean;
  //! the gradient of the GP mean evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_mu;
  //! the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points
  std::vector<double> cholesky_to_sample_var;
  //! the gradient of the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_chol_decomp;

  bool configure_for_test;
  //! structs containing pointers to store the memory locations of variables on GPU
  //! input data for GPU computations and GPU should not modify them
  CudaDevicePointer gpu_mu;
  CudaDevicePointer gpu_chol_var;
  CudaDevicePointer gpu_grad_mu;
  CudaDevicePointer gpu_grad_chol_var;
  //! data containing results returned by GPU computations
  CudaDevicePointer gpu_ei_storage;
  CudaDevicePointer gpu_grad_ei_storage;
  //! data containing random numbers used in GPU computations, which are only
  //! used for testing
  CudaDevicePointer gpu_random_number_ei;
  CudaDevicePointer gpu_random_number_grad_ei;

  //! storage for random numbers used in computing EI & grad_ei, this is only used to setup unit test
  std::vector<double> random_number_ei;
  std::vector<double> random_number_grad_ei;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementState);
};

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs).  Starts a GD run from each point in ``start_point_set``.  The point corresponding to the
  optimal EI\* is stored in ``best_next_point``.

  \* Multistarting is heuristic for global optimization. EI is not convex so this method may not find the true optimum.

  This function wraps MultistartOptimizer<>::MultistartOptimize() (see ``gpp_optimization.hpp``), which provides the multistarting
  component. Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``. Please see that file for details on gradient descent and see ``gpp_optimizer_parameters.hpp``
  for the meanings of the GradientDescentParameters.

  This function (or its wrappers, e.g., ComputeOptimalPointsToSampleWithRandomStarts) are the primary entry-points for
  gradient descent based EI optimization in the ``optimal_learning`` library.

  Users may prefer to call ComputeOptimalPointsToSample(), which applies other heuristics to improve robustness.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING::
       This function fails ungracefully if NO improvement can be found!  In that case,
       ``best_next_point`` will always be the first point in ``start_point_set``.
       ``found_flag`` will indicate whether this occured.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :start_point_set[dim][num_to_sample][num_multistarts]: set of initial guesses for MGD (one block of num_to_sample points per multistart)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points in set of initial guesses
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero EI
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void CudaComputeOptimalPointsToSampleViaMultistartGradientDescent(
    const GaussianProcess& gaussian_process,
    const GradientDescentParameters& optimizer_parameters,
    const DomainType& domain,
    const ThreadSchedule thread_schedule,
    double const * restrict start_point_set,
    double const * restrict points_being_sampled,
    int num_multistarts,
    int num_to_sample,
    int num_being_sampled,
    double best_so_far,
    int max_int_steps,
    UniformRandomGenerator* uniform_rng,
    bool * restrict found_flag,
    int which_gpu,
    double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  bool configure_for_gradients = true;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, thread_schedule.max_num_threads,
                                  configure_for_gradients, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, start_point_set);

    GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimizer_parameters,
                                            domain, thread_schedule, start_point_set,
                                            num_multistarts,
                                            ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    CudaExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far, which_gpu);

    typename CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, start_point_set, points_being_sampled,
                                                                  num_to_sample, num_being_sampled, configure_for_gradients,
                                                                  uniform_rng);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state.GetProblemSize(), 0.0, start_point_set);

    using RepeatedDomain = RepeatedDomain<DomainType>;
    RepeatedDomain repeated_domain(domain, num_to_sample);
    GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimizer_parameters,
                                            repeated_domain, thread_schedule, start_point_set,
                                            num_multistarts,
                                            &ei_state, nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs), starting from ``num_multistarts`` points selected randomly from the within th domain.

  This function is a simple wrapper around ComputeOptimalPointsToSampleViaMultistartGradientDescent(). It additionally
  generates a set of random starting points and is just here for convenience when better initial guesses are not
  available.

  See ComputeOptimalPointsToSampleViaMultistartGradientDescent() for more details.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSampleWithRandomStarts(const GaussianProcess& gaussian_process,
                                                      const GradientDescentParameters& optimizer_parameters,
                                                      const DomainType& domain, const ThreadSchedule& thread_schedule,
                                                      double const * restrict points_being_sampled,
                                                      int num_to_sample, int num_being_sampled, double best_so_far,
                                                      int max_int_steps, bool * restrict found_flag, int which_gpu,
                                                      UniformRandomGenerator * uniform_generator,
                                                      double * restrict best_next_point) {
  std::vector<double> starting_points(gaussian_process.dim()*optimizer_parameters.num_multistarts*num_to_sample);

  // GenerateUniformPointsInDomain() is allowed to return fewer than the requested number of multistarts
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  int num_multistarts = repeated_domain.GenerateUniformPointsInDomain(optimizer_parameters.num_multistarts,
                                                                      uniform_generator, starting_points.data());

  CudaComputeOptimalPointsToSampleViaMultistartGradientDescent(gaussian_process, optimizer_parameters, domain,
                                                               thread_schedule, starting_points.data(),
                                                               points_being_sampled, num_multistarts, num_to_sample,
                                                               num_being_sampled, best_so_far, max_int_steps,
                                                               uniform_generator, found_flag, which_gpu, best_next_point);
#ifdef OL_WARNING_PRINT
  if (false == *found_flag) {
    OL_WARNING_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
    OL_WARNING_PRINTF("First multistart point was returned:\n");
    PrintMatrixTrans(starting_points.data(), num_to_sample, gaussian_process.dim());
  }
#endif
}

/*!\rst
  Function to evaluate Expected Improvement (q,p-EI) over a specified list of ``num_multistarts`` points.
  Optionally outputs the EI at each of these points.
  Outputs the point of the set obtaining the maximum EI value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
  This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

  This function is just a wrapper that builds the required state objects and a NullOptimizer object and calls
  MultistartOptimizer<...>::MultistartOptimize(...); see gpp_optimization.hpp.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :initial_guesses[dim][num_to_sample][num_multistarts]: list of points at which to compute EI
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: the device ID of GPU used for computation
    :uniform_rng[1]: a UniformRandomGenerator object that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_rng[1]: UniformRandomGenerator object will have its state changed due to random draws
    :function_values[num_multistarts]: EI evaluated at each point of ``initial_guesses``, in the same order as
      ``initial_guesses``; never dereferenced if nullptr
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
void CudaEvaluateEIAtPointList(const GaussianProcess& gaussian_process,
                               const ThreadSchedule& thread_schedule,
                               double const * restrict initial_guesses,
                               double const * restrict points_being_sampled,
                               int num_multistarts, int num_to_sample,
                               int num_being_sampled, double best_so_far,
                               int max_int_steps, bool * restrict found_flag,
                               int which_gpu, UniformRandomGenerator* uniform_rng, 
                               double * restrict function_values,
                               double * restrict best_next_point);

/*!\rst
  Perform a random, naive search to "solve" the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs).  Evaluates EI at ``num_multistarts`` points (e.g., on a latin hypercube) to find the
  point with the best EI value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  Wraps EvaluateEIAtPointList(); constructs the input point list with a uniform random sampling from the given Domain object.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of random points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: device ID of the GPU used for computation
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(const GaussianProcess& gaussian_process,
                                                             const DomainType& domain,
                                                             const ThreadSchedule& thread_schedule,
                                                             double const * restrict points_being_sampled,
                                                             int num_multistarts, int num_to_sample,
                                                             int num_being_sampled, double best_so_far,
                                                             int max_int_steps, bool * restrict found_flag,
                                                             int which_gpu, UniformRandomGenerator * uniform_generator,
                                                             double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts*num_to_sample);
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  num_multistarts = repeated_domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator,
                                                                  initial_guesses.data());

  CudaEvaluateEIAtPointList(gaussian_process, thread_schedule, initial_guesses.data(),
                            points_being_sampled, num_multistarts, num_to_sample,
                            num_being_sampled, best_so_far, max_int_steps,
                            found_flag, which_gpu, uniform_generator, nullptr, best_next_point);
}

/*!\rst
  Solve the q,p-EI problem (see header docs) by optimizing the Expected Improvement.
  Uses multistart gradient descent, "dumb" search, and/or other heuristics to perform the optimization.

  This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
  improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

  Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
  experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
  (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
  (requires monte-carlo iteration), so this method is usually very expensive.

  Wraps ComputeOptimalPointsToSampleWithRandomStarts() and ComputeOptimalPointsToSampleViaLatinHypercubeSearch().

  Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
  makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
  GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

  .. NOTE:: These comments were copied into multistart_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
    :num_lhc_samples: number of samples to draw if/when doing latin hypercube search
    :which_gpu: device ID of GPU used for computation
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if best_points_to_sample corresponds to a nonzero EI if sampled simultaneously
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :best_points_to_sample[num_to_sample*dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSample(const GaussianProcess& gaussian_process,
                                      const GradientDescentParameters& optimizer_parameters,
                                      const DomainType& domain, const ThreadSchedule& thread_schedule,
                                      double const * restrict points_being_sampled,
                                      int num_to_sample, int num_being_sampled, double best_so_far,
                                      int max_int_steps, bool lhc_search_only,
                                      int num_lhc_samples, bool * restrict found_flag, int which_gpu,
                                      UniformRandomGenerator * uniform_generator,
                                      double * restrict best_points_to_sample);

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, int which_gpu,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps,
    bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag, int which_gpu,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

#endif  // OL_GPU_ENABLED

}   // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
