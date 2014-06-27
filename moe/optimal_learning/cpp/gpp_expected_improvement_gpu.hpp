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
#include "gpp_random.hpp"
#include "gpp_math.hpp"


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
      :ei_state[1]: state with temporary storage modified; ``normal_rng`` modified
    \return
      the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    This function has the same functionality as ComputeGradExpectedImprovement in class ExpectedImprovementEvaluator.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``normal_rng`` modified
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
      :normal_rng[1]: pointer to a properly initialized* NormalRNG object

    .. NOTE::
         * The NormalRNG object must already be seeded.  If multithreaded computation is used for EI, then every state object
         must have a different NormalRNG (different seeds, not just different objects).
  \endrst*/
  CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, NormalRNG * normal_rng_in);

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
  NormalRNG * normal_rng;

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
  //! structs containing pointers to store the memory locations of variables on GPU
  CudaDevicePointer gpu_mu;
  CudaDevicePointer gpu_L;
  CudaDevicePointer gpu_grad_mu;
  CudaDevicePointer gpu_grad_L;
  CudaDevicePointer gpu_EI_storage;
  CudaDevicePointer gpu_grad_EI_storage;
#endif

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(CudaExpectedImprovementState);
};

}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_HPP_
