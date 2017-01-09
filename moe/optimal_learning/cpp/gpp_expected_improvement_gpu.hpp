/*!
  \file gpp_expected_improvement_gpu.hpp
  \rst
  All GPU related functions are declared here, and any other C++ functions who wish to call GPU functions should only call functions here.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_

#ifdef OL_GPU_ENABLED
#include <driver_types.h>
#endif

#include <algorithm>
#include <memory>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
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
  A deleter for std::unique_ptr that are created with memory returned by ``cudaMalloc()``;
  e.g., the member of ``CudaDevicePointer``.

  For a description of deleters, see:
  http://en.cppreference.com/w/cpp/memory/unique_ptr
  http://en.cppreference.com/w/cpp/memory/unique_ptr/~unique_ptr

  The STL-provided default deleter:
  http://en.cppreference.com/w/cpp/memory/default_delete
\endrst*/
struct CudaDeleter final {
  /*!\rst
    Free the memory pointed to by ``device_ptr``.
    Called as part of the dtor for ``std::unique_ptr`` and MUST NOT throw exceptions.
    Wraps ``cudaFree()``.

    \param
      :device_ptr: device pointer to memory previously allocated by ``cudaMalloc()``.
  \endrst*/
  void operator()(void * device_ptr) const noexcept;
};

/*!\rst
  This struct is a smart pointer that wraps ``std::unique_ptr``. It provides a simple
  interface for device memory (i.e., on a GPU) ownership. It automatically handles
  ``cudaMalloc()`` and ``cudaFree()`` calls and error checks; the result is stored
  in a ``std::unique_ptr``.
\endrst*/
template <typename ValueType>
class CudaDevicePointer final {
 public:
  /*!\rst
    Construct a CudaDevicePointer (via ``cudaMalloc()``) that *owns* a block of
    ``num_values * sizeof(ValueType)`` bytes on the GPU device.

    If allocation fails, device_ptr is nullptr and num_values is 0.

    \param
      :num_values: number of values allocated at the device memory address held in this object
  \endrst*/
  explicit CudaDevicePointer(int num_values_in);

  CudaDevicePointer(CudaDevicePointer&& other);

  int num_values() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_values_;
  }

  ValueType * device_ptr() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return device_ptr_.get();
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaDevicePointer);

 private:
  //! number of values to allocate on gpu, so the memory size is ``num_values * sizeof(ValueType)``
  int num_values_;
  //! pointer to the memory location on gpu
  std::unique_ptr<ValueType, CudaDeleter> device_ptr_;
};

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template class CudaDevicePointer<int>;
extern template class CudaDevicePointer<double>;

/*!\rst
  Exception to handle runtime errors returned by CUDA API functions. This class subclasses
  OptimalLearningException in gpp_exception.hpp/cpp, and basically has the same functionality
  as its superclass, except the constructor is different.
\endrst*/
class OptimalLearningCudaException : public OptimalLearningException {
 public:
  //! String name of this exception ofr logging.
  constexpr static char const * kName = "OptimalLearningCudaException";

  /*!\rst
    Constructs a OptimalLearningCudaException with struct CudaError.

    \param
      :error: C struct that contains error message returned by CUDA API functions
  \endrst*/
  explicit OptimalLearningCudaException(const CudaError& error);

  OL_DISALLOW_DEFAULT_AND_ASSIGN(OptimalLearningCudaException);
};

struct CudaExpectedImprovementState;

/*!\rst
  This class has the same functionality as ``ExpectedImprovementEvaluator`` (see ``gpp_math.hpp``),
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
  This has the same functionality as ``ExpectedImprovementState`` (see ``gpp_math.hpp``) except that it is for GPU computing
\endrst*/
struct CudaExpectedImprovementState final {
  using EvaluatorType = CudaExpectedImprovementEvaluator;

  /*!\rst
    This struct has same functionality as ``ExpectedImprovementState`` in ``gpp_math.hpp``,
    except that it is specifically for GPU EI evaluator. Refer to ``gpp_math.hpp`` for detailed
    documentation.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :points_to_sample[dim][num_to_sample]: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :points_being_sampled[dim][num_being_sampled]: points being sampled in concurrent experiments
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
      :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
      :uniform_rng[1]: pointer to a properly initialized\* UniformRandomGenerator object

    .. NOTE::
         \* The UniformRandomGenerator object must already be seeded.  If multithreaded computation is used for EI, then every state object
         must have a different UniformRandomGenerator (different seeds, not just different objects).
  \endrst*/
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                               double const * restrict points_being_sampled, int num_to_sample_in,
                               int num_being_sampled_in, bool configure_for_gradients,
                               UniformRandomGenerator * uniform_rng_in);

  // constructor for setting up unit test
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                               double const * restrict points_being_sampled, int num_to_sample_in,
                               int num_being_sampled_in, bool configure_for_gradients,
                               UniformRandomGenerator * uniform_rng_in, bool configure_for_test);

  CudaExpectedImprovementState(CudaExpectedImprovementState&& other);

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
  UniformRandomGenerator * uniform_rng;

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
  //! structs wrapping GPU device pointers used to store GP quantities
  //! temp device space for input data for GPU computations; GPU should not modify these
  CudaDevicePointer<double> gpu_mu;
  CudaDevicePointer<double> gpu_grad_mu;
  CudaDevicePointer<double> gpu_chol_var;
  CudaDevicePointer<double> gpu_grad_chol_var;

  //! temp device space for intermediate results returned by GPU computations
  CudaDevicePointer<double> gpu_ei_storage;
  CudaDevicePointer<double> gpu_grad_ei_storage;

  //! device storage for random numbers used by the GPU computing of ei & grad_ei
  //! only used for testing
  CudaDevicePointer<double> gpu_random_number_ei;
  CudaDevicePointer<double> gpu_random_number_grad_ei;

  //! host storage for random numbers used by the GPU in computing ei & grad_ei
  //! only used for testing
  std::vector<double> random_number_ei;
  std::vector<double> random_number_grad_ei;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementState);
};

/*!\rst
  Set up vector of CudaExpectedImprovementEvaluator::StateType.

  This is a utility function just for reducing code duplication.

  Throws ``InvalidValueException`` if ``max_num_threads != 1``. Multiple threads (=> multiple GPUs)
  is not yet supported.
  TODO(GH-398): remove this requirement/comments when we support computation on multiple GPUs

  \param
    :ei_evaluator: evaluator object associated w/the state objects being constructed
    :points_to_sample[dim][num_to_sample]: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrently experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
    :uniform_rng[1]: UniformRandomGenerator object used to seed the GPU PRNG(s)
  \output
    :uniform_rng[1]: UniformRandomGenerator object will have its state changed due to random draws
    :state_vector[max_num_threads]: vector of states containing ``max_num_threads`` properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupExpectedImprovementState(
    const CudaExpectedImprovementEvaluator& ei_evaluator,
    double const * restrict points_to_sample,
    double const * restrict points_being_sampled,
    int num_to_sample,
    int num_being_sampled,
    int max_num_threads,
    bool configure_for_gradients,
    UniformRandomGenerator * uniform_rng,
    std::vector<typename CudaExpectedImprovementEvaluator::StateType> * state_vector) {
  // TODO(GH-398): remove this requirement when we support computation on multiple GPUs
  if (unlikely(max_num_threads != 1)) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "max_num_threads must equal to 1 when using GPU functions!", max_num_threads, 1);
  }

  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(ei_evaluator, points_to_sample, points_being_sampled, num_to_sample,
                               num_being_sampled, configure_for_gradients, uniform_rng + i);
  }
}

/*!\rst
  This function is the same as ``ComputeOptimalPointsToSampleViaMultistartGradientDescent`` in ``gpp_math.hpp`` except that it is
  specifically used for GPU EI evaluators. Refer to ``gpp_math.hpp`` for detailed documentation.

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
    :which_gpu: the device ID of GPU used for computation
    :uniform_rng[1]: UniformRandomGenerator object used to seed the GPU PRNG(s)
  \output
    :uniform_rng[1]: UniformRandomGenerator object will have its state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero EI
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void CudaComputeOptimalPointsToSampleViaMultistartGradientDescent(
    const GaussianProcess& gaussian_process,
    const GradientDescentParameters& optimizer_parameters,
    const DomainType& domain,
    const ThreadSchedule& thread_schedule,
    double const * restrict start_point_set,
    double const * restrict points_being_sampled,
    int num_multistarts,
    int num_to_sample,
    int num_being_sampled,
    double best_so_far,
    int max_int_steps,
    int which_gpu,
    UniformRandomGenerator * uniform_rng,
    bool * restrict found_flag,
    double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  bool configure_for_gradients = true;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    ComputeOptimalPointsToSampleViaMultistartGradientDescent(gaussian_process, optimizer_parameters, domain, thread_schedule,
                                                             start_point_set, points_being_sampled, num_multistarts, num_to_sample,
                                                             num_being_sampled, best_so_far, max_int_steps, nullptr, found_flag,
                                                             best_next_point);
  } else {
    CudaExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far, which_gpu);

    std::vector<typename CudaExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, points_being_sampled, num_to_sample,
                                  num_being_sampled, thread_schedule.max_num_threads,
                                  configure_for_gradients, uniform_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, start_point_set);

    using RepeatedDomain = RepeatedDomain<DomainType>;
    RepeatedDomain repeated_domain(domain, num_to_sample);
    GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimizer_parameters,
                                            repeated_domain, thread_schedule, start_point_set,
                                            num_multistarts,
                                            ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  This function is the same as ``ComputeOptimalPointsToSampleWithRandomStarts`` in ``gpp_math.hpp`` except that it is
  specifically used for GPU EI evaluators. Refer to ``gpp_math.hpp`` for detailed documentation.

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
    :which_gpu: the device ID of GPU used for computation
    :uniform_generator[1]: UniformRandomGenerator object used to seed the GPU PRNG(s)
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSampleWithRandomStarts(const GaussianProcess& gaussian_process,
                                                      const GradientDescentParameters& optimizer_parameters,
                                                      const DomainType& domain, const ThreadSchedule& thread_schedule,
                                                      double const * restrict points_being_sampled,
                                                      int num_to_sample, int num_being_sampled, double best_so_far,
                                                      int max_int_steps, int which_gpu, bool * restrict found_flag,
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
                                                               num_being_sampled, best_so_far, max_int_steps, which_gpu,
                                                               uniform_generator, found_flag, best_next_point);
#ifdef OL_WARNING_PRINT
  if (false == *found_flag) {
    OL_WARNING_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
    OL_WARNING_PRINTF("First multistart point was returned:\n");
    PrintMatrixTrans(starting_points.data(), num_to_sample, gaussian_process.dim());
  }
#endif
}

/*!\rst
  This function is the same as ``EvaluateEIAtPointList`` in ``gpp_math.hpp`` except that it is
  specifically used for GPU EI evaluators. Refer to ``gpp_math.hpp`` for detailed documentation.

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
    :uniform_rng[1]: UniformRandomGenerator object used to seed the GPU PRNG(s)
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
                               int max_int_steps, int which_gpu, bool * restrict found_flag,
                               UniformRandomGenerator * uniform_rng,
                               double * restrict function_values,
                               double * restrict best_next_point);

/*!\rst
  This function is the same as ``ComputeOptimalPointsToSampleViaLatinHypercubeSearch`` in ``gpp_math.hpp`` except that it is
  specifically used for GPU EI evaluators. Refer to ``gpp_math.hpp`` for detailed documentation.

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
                                                             int max_int_steps, int which_gpu, bool * restrict found_flag,
                                                             UniformRandomGenerator * uniform_generator,
                                                             double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts*num_to_sample);
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  num_multistarts = repeated_domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator,
                                                                  initial_guesses.data());

  CudaEvaluateEIAtPointList(gaussian_process, thread_schedule, initial_guesses.data(),
                            points_being_sampled, num_multistarts, num_to_sample,
                            num_being_sampled, best_so_far, max_int_steps, which_gpu,
                            found_flag, uniform_generator, nullptr, best_next_point);
}

/*!\rst
  This function is the same as ``ComputeOptimalPointsToSample`` in ``gpp_math.hpp`` except that it is
  specifically used for GPU EI evaluators. Refer to ``gpp_math.hpp`` for detailed documentation.

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
    :uniform_generator[1]: UniformRandomGenerator object used to seed the GPU PRNG(s)
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
                                      int num_lhc_samples, int which_gpu, bool * restrict found_flag,
                                      UniformRandomGenerator * uniform_generator,
                                      double * restrict best_points_to_sample);

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, int which_gpu, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps,
    bool lhc_search_only, int num_lhc_samples, int which_gpu, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

#endif  // OL_GPU_ENABLED

}   // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
