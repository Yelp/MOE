/*!
  \file gpp_expected_improvement_gpu.hpp
  \rst
  All gpu related functions are declared here, and any other cpp functions who wish to use gpu implementation should only call functions here.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_

#include <vector>
#include <algorithm>

#include "gpp_common.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"


namespace optimal_learning {
/*!\rst
  This struct contains pointer to memory location on GPU, its constructor and destructor also take care of memory allocation/deallocation on GPU. When GPU is not enabled, it becomes some dummy struct.
\endrst*/
#ifdef OL_GPU_ENABLED
struct CudaDevicePointer final {
  explicit CudaDevicePointer(int num_doubles_in);
  ~CudaDevicePointer();

  //! pointer to the memory location on gpu
  double* ptr;
  //! number of doubles to allocate on gpu, so the memory size is num_doubles * sizeof(double)
  int num_doubles;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaDevicePointer);
};
#endif

struct CudaExpectedImprovementState;

/*!\rst
  This class has the same functionality as ExpectedImprovementEvaluator, except that computations are performed on GPU.
\endrst*/
class CudaExpectedImprovementEvaluator final {
 public:
  using StateType = CudaExpectedImprovementState;
  /*!\rst
    Constructs a CudaExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :num_mc_iterations: number of monte carlo iterations
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  CudaExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, int num_mc_in, double best_so_far) : dim_(gaussian_process_in.dim()),  num_mc(num_mc_in), best_so_far_(best_so_far), gaussian_process_(&gaussian_process_in) {
    setupGPU(0);
  }

  /*!\rst
    Constructor that also specify which gpu you want to use (for multi-gpu system)
  \endrst*/
  CudaExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, int num_mc_in, double best_so_far, int devID_in) : dim_(gaussian_process_in.dim()), num_mc(num_mc_in), best_so_far_(best_so_far), gaussian_process_(&gaussian_process_in) {
    setupGPU(devID_in);
  }

  ~CudaExpectedImprovementEvaluator();

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_mc_itr() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_mc;
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
  void ComputeGradObjectiveFunction(StateType * ei_state, double * restrict grad_EI) const OL_NONNULL_POINTERS {
    ComputeGradExpectedImprovement(ei_state, grad_EI);
  }

  /*!\rst
    This function has the same functionality as ComputeExpectedImprovement in class ExpectedImprovementEvaluator.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``uniform_rng`` modified
    \return
      the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    This function has the same functionality as ComputeGradExpectedImprovement in class ExpectedImprovementEvaluator.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``uniform_rng`` modified
      :grad_EI[dim][num_to_sample]: gradient of EI
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const OL_NONNULL_POINTERS;

  void setupGPU(int devID);

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! number of mc iterations
  int num_mc;
  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;
  //! pointer to gaussian process used in EI computations
  const GaussianProcess * gaussian_process_;
};

/*!\rst
  This has the same functionality as ExpectedImprovementState except that it is for GPU computing
\endrst*/
struct CudaExpectedImprovementState final {
  using EvaluatorType = CudaExpectedImprovementEvaluator;

  /*!\rst
    Constructs an CudaExpectedImprovementState object with a specified source of randomness for the purpose of computing EI
    (and its gradient) over the specified set of points to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ei_evaluator).

    .. WARNING:: This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.

    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in UNDEFINED BEHAVIOR.

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
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, UniformRandomGenerator* uniform_rng_in);

#ifdef OL_GPU_ENABLED
  // constructor for setting up unit test
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, UniformRandomGenerator * uniform_rng_in, bool configure_for_test);
#endif

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
  static std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample, int num_being_sampled, int dim) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> union_of_points(dim*(num_to_sample + num_being_sampled));
    std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.data());
    std::copy(points_being_sampled, points_being_sampled + dim*num_being_sampled, union_of_points.data() + dim*num_to_sample);
    return union_of_points;
  }

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
  void UpdateCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) OL_NONNULL_POINTERS {
    // update points_to_sample in union_of_points
    std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

    // evaluate derived quantities for the GP
    points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives);
  }

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
  void SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) OL_NONNULL_POINTERS {
    // update quantities derived from points_to_sample
    UpdateCurrentPoint(ei_evaluator, points_to_sample);
  }

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

#ifdef OL_GPU_ENABLED
  bool configure_for_test;
  //! structs containing pointers to store the memory locations of variables on GPU
  CudaDevicePointer gpu_mu;
  CudaDevicePointer gpu_L;
  CudaDevicePointer gpu_grad_mu;
  CudaDevicePointer gpu_grad_L;
  CudaDevicePointer gpu_EI_storage;
  CudaDevicePointer gpu_grad_EI_storage;
  CudaDevicePointer gpu_random_number_EI;
  CudaDevicePointer gpu_random_number_gradEI;

  //! storage for random numbers used in computing EI & grad_EI, this is only used to setup unit test
  std::vector<double> random_number_EI;
  std::vector<double> random_number_gradEI;
#endif

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementState);
};


/*!\rst
  This function is the same as ComputeOptimalPointsToSampleViaMultistartGradientDescent in gpp_math.hpp, except that it uses 
  GPU for MC simulation.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :start_point_set[dim][num_to_sample][num_multistarts]: set of initial guesses for MGD (one block of num_to_sample points per multistart)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points in set of initial guesses
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: ID of gpu to use
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores), this is used for 1,0-EI only
    :uniform_rng[1]: UniformRandomGenerator object that provides the initial seed for gpu computation
  \output
    :uniform_rng[1]: UniformRandomGenerator object will have its state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero EI
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void CudaComputeOptimalPointsToSampleViaMultistartGradientDescent(
    const GaussianProcess& gaussian_process,
    const GradientDescentParameters& optimization_parameters,
    const DomainType& domain,
    double const * restrict start_point_set,
    double const * restrict points_being_sampled,
    int num_multistarts,
    int num_to_sample,
    int num_being_sampled,
    double best_so_far,
    int max_int_steps,
    int which_gpu,
    int max_num_threads,
    UniformRandomGenerator* uniform_rng,
    bool * restrict found_flag,
    double * restrict best_next_point) {
  // set chunk_size; see gpp_common.hpp header comments, item 7
  const int chunk_size = std::max(std::min(4, std::max(1, num_multistarts/max_num_threads)),
                                  num_multistarts/(max_num_threads*10));

  bool configure_for_gradients = true;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, max_num_threads,
                                  configure_for_gradients, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, start_point_set);

    GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimization_parameters,
                                            domain, start_point_set, num_multistarts,
                                            max_num_threads, chunk_size, ei_state_vector.data(),
                                            nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    CudaExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far, which_gpu);

    typename CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, start_point_set, points_being_sampled, num_to_sample, num_being_sampled, configure_for_gradients, uniform_rng);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state.GetProblemSize(), 0.0, start_point_set);

    using RepeatedDomain = RepeatedDomain<DomainType>;
    RepeatedDomain repeated_domain(domain, num_to_sample);
    GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<CudaExpectedImprovementEvaluator, RepeatedDomain> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimization_parameters,
                                            repeated_domain, start_point_set, num_multistarts,
                                            1, 1, &ei_state,
                                            nullptr, &io_container);
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
    :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: ID of gpu to use
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores), this is used for 1,0-EI only
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSampleWithRandomStarts(const GaussianProcess& gaussian_process,
                                                  const GradientDescentParameters& optimization_parameters,
                                                  const DomainType& domain,
                                                  double const * restrict points_being_sampled,
                                                  int num_to_sample, int num_being_sampled, double best_so_far,
                                                  int max_int_steps, int which_gpu, int max_num_threads, 
                                                  bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
                                                  double * restrict best_next_point) {
  std::vector<double> starting_points(gaussian_process.dim()*optimization_parameters.num_multistarts*num_to_sample);

  // GenerateUniformPointsInDomain() is allowed to return fewer than the requested number of multistarts
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  int num_multistarts = repeated_domain.GenerateUniformPointsInDomain(optimization_parameters.num_multistarts,
                                                                      uniform_generator, starting_points.data());

  CudaComputeOptimalPointsToSampleViaMultistartGradientDescent(gaussian_process, optimization_parameters, domain,
                                                           starting_points.data(), points_being_sampled,
                                                           num_multistarts, num_to_sample, num_being_sampled,
                                                           best_so_far, max_int_steps, which_gpu, max_num_threads,
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
  This function is the same as EvaluateEIAtPointList in gpp_math.hpp, except for MC computation it uses GPU for evaluation.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :initial_guesses[dim][num_to_sample][num_multistarts]: list of points at which to compute EI
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: ID of gpu to use
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores), this is used for 1,0-EI only
    :uniform_rng[1]: UniformRandomGenerator object that provides initial seed for computation on GPU
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_rng[1]: UniformRandomGenerator object will have their state changed due to random draws
    :function_values[num_multistarts]: EI evaluated at each point of ``initial_guesses``, in the same order as
      ``initial_guesses``; never dereferenced if nullptr
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
void CudaEvaluateEIAtPointList(const GaussianProcess& gaussian_process, double const * restrict initial_guesses,
                           double const * restrict points_being_sampled, int num_multistarts, int num_to_sample,
                           int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu, int max_num_threads,
                           bool * restrict found_flag, UniformRandomGenerator* uniform_rng, double * restrict function_values,
                           double * restrict best_next_point); 

/*!\rst
  This function is the same as ComputeOptimalPointsToSampleViaLatinHypercubeSearch in gpp_math.hpp, except that it uses GPU
  to compute Expected Improvement.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of random points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: ID of GPU to use
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores), this is used for 1,0-EI only
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(const GaussianProcess& gaussian_process,
                                                         const DomainType& domain,
                                                         double const * restrict points_being_sampled,
                                                         int num_multistarts, int num_to_sample,
                                                         int num_being_sampled, double best_so_far,
                                                         int max_int_steps, int which_gpu, int max_num_threads,
                                                         bool * restrict found_flag,
                                                         UniformRandomGenerator * uniform_generator,
                                                         double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts*num_to_sample);
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  num_multistarts = repeated_domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator,
                                                                  initial_guesses.data());

  CudaEvaluateEIAtPointList(gaussian_process, initial_guesses.data(), points_being_sampled, num_multistarts,
                        num_to_sample, num_being_sampled, best_so_far, max_int_steps, which_gpu, max_num_threads,
                        found_flag, uniform_generator, nullptr, best_next_point);
}

/*!\rst
  This function is virtually the same as ComputeOptimalPointsToSample in gpp_math.hpp, except it specifically uses gpu
  to compute Expected Improvement.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :which_gpu: ID of gpu to use for computation
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores), this is used for 1,0-EI only
    :lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
    :num_lhc_samples: number of samples to draw if/when doing latin hypercube search
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[1]: a NormalRNG objects that provide the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_points_to_sample corresponds to a nonzero EI if sampled simultaneously
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[1]: NormalRNG object will have its state changed due to random draws
    :best_points_to_sample[num_to_sample*dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSample(const GaussianProcess& gaussian_process,
                                  const GradientDescentParameters& optimization_parameters,
                                  const DomainType& domain, double const * restrict points_being_sampled,
                                  int num_to_sample, int num_being_sampled, double best_so_far,
                                  int max_int_steps, int which_gpu, int max_num_threads, bool lhc_search_only,
                                  int num_lhc_samples, bool * restrict found_flag,
                                  UniformRandomGenerator * uniform_generator,
                                  double * restrict best_points_to_sample);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters,
    const TensorProductDomain& domain, double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu, int max_num_threads, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    double * restrict best_points_to_sample);
extern template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters,
    const SimplexIntersectTensorProductDomain& domain, double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu,
    int max_num_threads, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
