/*!
  \file gpp_expected_improvement_gpu.cpp
  \rst
  gpu code for calculating Expected Improvemnet & gradient of Expected Improvement
\endrst*/

#include "gpp_expected_improvement_gpu.hpp"

#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"

#ifdef OL_GPU_ENABLED
#include "gpu/gpp_cuda_math.hpp"
#include "driver_types.h"
#include "cuda_runtime.h"

#define OL_CUDA_THROW_EXCEPTION(_ERR) ThrowException(OptimalLearningException((_ERR).line_info, (_ERR).func_info, cudaGetErrorString((_ERR).err)));
#define OL_CUDA_ERROR_THROW(_ERR) {if((_ERR).err != cudaSuccess) {OL_CUDA_THROW_EXCEPTION(_ERR)}}
#endif

namespace optimal_learning {
#ifdef OL_GPU_ENABLED
CudaDevicePointer::CudaDevicePointer(int num_doubles_in)
  : num_doubles(num_doubles_in) {
    if (num_doubles_in > 0) {
        CudaError _err = cuda_allocate_mem_for_double_vector(num_doubles, &ptr);
        if (_err.err != cudaSuccess) {
            ptr = nullptr;
            OL_CUDA_THROW_EXCEPTION(_err)
        }
    } else {
        ptr = nullptr;
    }
  }

CudaDevicePointer::~CudaDevicePointer() {
  if (ptr != nullptr) {
      cuda_free_mem(ptr);
  }
}

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double EI_val;
  int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.", ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }
  unsigned int seed_in = (ei_state->uniform_rng->GetEngine())();
  CudaError _err = cuda_get_EI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(), best_so_far_, num_union, ei_state->gpu_mu.ptr, ei_state->gpu_L.ptr, ei_state->gpu_EI_storage.ptr, seed_in, num_mc, &EI_val, ei_state->gpu_random_number_EI.ptr, ei_state->random_number_EI.data(), ei_state->configure_for_test);
  OL_CUDA_ERROR_THROW(_err)
  return EI_val;
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  if (ei_state->num_derivatives == 0) {
    OL_THROW_EXCEPTION(OptimalLearningException, "configure_for_gradients set to false, gradient computation is disabled!");
  }
  const int num_union = ei_state->num_union;
  const int num_to_sample = ei_state->num_to_sample;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, ei_state->grad_mu.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.", ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }

  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data());
  unsigned int seed_in = (ei_state->uniform_rng->GetEngine())();

  CudaError _err = cuda_get_gradEI(ei_state->to_sample_mean.data(), ei_state->grad_mu.data(), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data(), best_so_far_, num_union, num_to_sample, dim_, (ei_state->gpu_mu).ptr, (ei_state->gpu_grad_mu).ptr, (ei_state->gpu_L).ptr, (ei_state->gpu_grad_L).ptr, (ei_state->gpu_grad_EI_storage).ptr, seed_in, num_mc, grad_EI, ei_state->gpu_random_number_gradEI.ptr, ei_state->random_number_gradEI.data(), ei_state->configure_for_test);
  OL_CUDA_ERROR_THROW(_err)
}

void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
  CudaError _err = cuda_set_device(devID);
  OL_CUDA_ERROR_THROW(_err)
}

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
  cudaDeviceReset();
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, UniformRandomGenerator * uniform_rng_in)
  : dim(ei_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
    uniform_rng(uniform_rng_in),
    to_sample_mean(num_union),
    grad_mu(dim*num_derivatives),
    cholesky_to_sample_var(Square(num_union)),
    grad_chol_decomp(dim*Square(num_union)*num_derivatives),
    configure_for_test(false),
    gpu_mu(num_union),
    gpu_L(Square(num_union)),
    gpu_grad_mu(dim * num_derivatives),
    gpu_grad_L(dim * Square(num_union) * num_derivatives),
    gpu_EI_storage(OL_EI_THREAD_NO * OL_EI_BLOCK_NO),
    gpu_grad_EI_storage(OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO * dim * num_derivatives),
    gpu_random_number_EI(0),
    gpu_random_number_gradEI(0),
    random_number_EI(0),
    random_number_gradEI(0) {
    }

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, UniformRandomGenerator * uniform_rng_in, bool configure_for_test_in)
  : dim(ei_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
    uniform_rng(uniform_rng_in),
    to_sample_mean(num_union),
    grad_mu(dim*num_derivatives),
    cholesky_to_sample_var(Square(num_union)),
    grad_chol_decomp(dim*Square(num_union)*num_derivatives),
    configure_for_test(configure_for_test_in),
    gpu_mu(num_union),
    gpu_L(Square(num_union)),
    gpu_grad_mu(dim * num_derivatives),
    gpu_grad_L(dim * Square(num_union) * num_derivatives),
    gpu_EI_storage(OL_EI_THREAD_NO * OL_EI_BLOCK_NO),
    gpu_grad_EI_storage(OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO * dim * num_derivatives),
    gpu_random_number_EI(configure_for_test ? (static_cast<int>(ei_evaluator.num_mc_itr() / (OL_EI_THREAD_NO * OL_EI_BLOCK_NO)) + 1) * (OL_EI_THREAD_NO * OL_EI_BLOCK_NO) * num_union : 0),
    gpu_random_number_gradEI(configure_for_test ? (static_cast<int>(ei_evaluator.num_mc_itr() / (OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO)) + 1) * (OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO) * num_union : 0),
    random_number_EI(configure_for_test ? (static_cast<int>(ei_evaluator.num_mc_itr() / (OL_EI_THREAD_NO * OL_EI_BLOCK_NO)) + 1) * (OL_EI_THREAD_NO * OL_EI_BLOCK_NO) * num_union : 0),
    random_number_gradEI(configure_for_test ? (static_cast<int>(ei_evaluator.num_mc_itr() / (OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO)) + 1) * (OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO) * num_union : 0) {
    }

#else

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, UniformRandomGenerator * uniform_rng_in)
  : dim(ei_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
    uniform_rng(uniform_rng_in),
    to_sample_mean(num_union),
    grad_mu(dim*num_derivatives),
    cholesky_to_sample_var(Square(num_union)),
    grad_chol_decomp(dim*Square(num_union)*num_derivatives) {
        OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
    }
#endif


/*!\rst
  Routes the EI computation through MultistartOptimizer + NullOptimizer to perform EI function evaluations at the list of input
  points, using the appropriate EI evaluator (e.g., monte carlo vs analytic) depending on inputs.
\endrst*/
void CudaEvaluateEIAtPointList(const GaussianProcess& gaussian_process, double const * restrict initial_guesses,
                           double const * restrict points_being_sampled, int num_multistarts, int num_to_sample,
                           int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu, int max_num_threads,
                           bool * restrict found_flag, UniformRandomGenerator* uniform_rng, double * restrict function_values,
                           double * restrict best_next_point) {
  // set chunk_size; see gpp_common.hpp header comments, item 7
  const int chunk_size = std::max(std::min(40, std::max(1, num_multistarts/max_num_threads)),
                                  num_multistarts/(max_num_threads*120));

  using DomainType = DummyDomain;
  DomainType dummy_domain;
  bool configure_for_gradients = false;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, max_num_threads,
                                  configure_for_gradients, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, initial_guesses);

    NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, dummy_domain,
                                            initial_guesses, num_multistarts, max_num_threads, chunk_size,
                                            ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    CudaExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far, which_gpu);

    typename CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, initial_guesses, points_being_sampled, num_to_sample, num_being_sampled, configure_for_gradients, uniform_rng);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state.GetProblemSize(), 0.0, initial_guesses);

    NullOptimizer<CudaExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<CudaExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<CudaExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, dummy_domain,
                                            initial_guesses, num_multistarts, 1, 1,
                                            &ei_state, function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  This is a simple wrapper around CudaComputeOptimalPointsToSampleWithRandomStarts() and
  ComputeOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).
\endrst*/
template <typename DomainType>
void CudaComputeOptimalPointsToSample(const GaussianProcess& gaussian_process,
                                  const GradientDescentParameters& optimization_parameters,
                                  const DomainType& domain, double const * restrict points_being_sampled,
                                  int num_to_sample, int num_being_sampled, double best_so_far,
                                  int max_int_steps, int which_gpu, int max_num_threads, bool lhc_search_only,
                                  int num_lhc_samples, bool * restrict found_flag,
                                  UniformRandomGenerator * uniform_generator,
                                  double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }

  std::vector<double> next_points_to_sample(gaussian_process.dim()*num_to_sample);

  bool found_flag_local = false;
  if (lhc_search_only == false) {
    CudaComputeOptimalPointsToSampleWithRandomStarts(gaussian_process, optimization_parameters, domain,
                                                 points_being_sampled, num_to_sample, num_being_sampled,
                                                 best_so_far, max_int_steps, which_gpu, max_num_threads,
                                                 &found_flag_local, uniform_generator,
                                                 next_points_to_sample.data());
  }

  // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
  if (found_flag_local == false || lhc_search_only == true) {
    if (unlikely(lhc_search_only == false)) {
      OL_WARNING_PRINTF("WARNING: %d,%d-EI opt DID NOT CONVERGE\n", num_to_sample, num_being_sampled);
      OL_WARNING_PRINTF("Attempting latin hypercube search\n");
    }

    CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, domain, points_being_sampled,
                                                        num_lhc_samples, num_to_sample, num_being_sampled,
                                                        best_so_far, max_int_steps, which_gpu, max_num_threads,
                                                        &found_flag_local, uniform_generator,
                                                        next_points_to_sample.data());

    // if latin hypercube 'dumb' search failed
    if (unlikely(found_flag_local == false)) {
      OL_ERROR_PRINTF("ERROR: %d,%d-EI latin hypercube search FAILED on\n", num_to_sample, num_being_sampled);
    }
  }

  // set outputs
  *found_flag = found_flag_local;
  std::copy(next_points_to_sample.begin(), next_points_to_sample.end(), best_points_to_sample);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters,
    const TensorProductDomain& domain, double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu, int max_num_threads, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    double * restrict best_points_to_sample);
template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters,
    const SimplexIntersectTensorProductDomain& domain, double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps, int which_gpu,
    int max_num_threads, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

}  // end namespace optimal_learning

