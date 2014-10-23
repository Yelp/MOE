/*!
  \file gpp_expected_improvement_gpu.cpp
  \rst
  This file contains implementations of GPU related functions. They are actually C++ wrappers for
  CUDA C functions defined in gpu/gpp_cuda_math.cu.
\endrst*/

#include "gpp_expected_improvement_gpu.hpp"

#include <cstdint>

#ifdef OL_GPU_ENABLED
#include <driver_types.h>
#include <cuda_runtime_api.h>
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
#endif

namespace optimal_learning {

#ifdef OL_GPU_ENABLED

void CudaDeleter::operator()(void * device_ptr) const noexcept {
  CudaError error = CudaFreeDeviceMemory(device_ptr);
  if (unlikely(error.err != cudaSuccess)) {
    // Throwing an exception out of a destructor is dangerous:
    // http://stackoverflow.com/questions/130117/throwing-exceptions-out-of-a-destructor
    // And this deleter functor serves as part of the dtor for std::unique_ptr:
    // http://en.cppreference.com/w/cpp/memory/unique_ptr/~unique_ptr

    // we want the formatted status message
    OptimalLearningCudaException cuda_failure(error);
    OL_ERROR_PRINTF("cudaFree error: %s\n", cuda_failure.what());
  }
}

template <typename ValueType>
CudaDevicePointer<ValueType>::CudaDevicePointer(int num_values_in) : num_values_(0) {
  if (num_values_in > 0) {
    ValueType * ptr;
    CudaError error = CudaMallocDeviceMemory(num_values_in * sizeof(ValueType), reinterpret_cast<void**>(&ptr));
    if (unlikely(error.err != cudaSuccess)) {
      ThrowException(OptimalLearningCudaException(error));
    } else {
      // store allocated memory and memory size only after allocation succeeds
      device_ptr_.reset(ptr);
      num_values_ = num_values_in;
    }
  }
}

template <typename ValueType>
CudaDevicePointer<ValueType>::CudaDevicePointer(CudaDevicePointer&& OL_UNUSED(other)) = default;

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
template class CudaDevicePointer<int>;
template class CudaDevicePointer<double>;

OptimalLearningCudaException::OptimalLearningCudaException(const CudaError& error)
      : OptimalLearningException(error.file_and_line_info, error.func_info, cudaGetErrorString(error.err)) {
}

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double EI_val;
  int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException,
                       "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
                       ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }
  uint64_t seed_in = (ei_state->uniform_rng->GetEngine())();
  OL_CUDA_ERROR_THROW(CudaGetEI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(),
                                num_union, num_mc_, best_so_far_,
                                seed_in, ei_state->configure_for_test,
                                ei_state->gpu_mu.device_ptr(),
                                ei_state->gpu_chol_var.device_ptr(),
                                ei_state->random_number_ei.data(),
                                ei_state->gpu_random_number_ei.device_ptr(),
                                ei_state->gpu_ei_storage.device_ptr(),
                                &EI_val));
  return EI_val;
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state,
                                                                      double * restrict grad_ei) const {
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
    OL_THROW_EXCEPTION(SingularMatrixException,
                       "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
                       ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }

  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state),
                                                         ei_state->cholesky_to_sample_var.data(),
                                                         ei_state->grad_chol_decomp.data());
  uint64_t seed_in = (ei_state->uniform_rng->GetEngine())();

  OL_CUDA_ERROR_THROW(CudaGetGradEI(ei_state->to_sample_mean.data(), ei_state->grad_mu.data(),
                                    ei_state->cholesky_to_sample_var.data(),
                                    ei_state->grad_chol_decomp.data(), num_union,
                                    num_to_sample, dim_, num_mc_, best_so_far_, seed_in,
                                    ei_state->configure_for_test,
                                    ei_state->gpu_mu.device_ptr(),
                                    ei_state->gpu_grad_mu.device_ptr(),
                                    ei_state->gpu_chol_var.device_ptr(),
                                    ei_state->gpu_grad_chol_var.device_ptr(),
                                    ei_state->random_number_grad_ei.data(),
                                    ei_state->gpu_random_number_grad_ei.device_ptr(),
                                    ei_state->gpu_grad_ei_storage.device_ptr(),
                                    grad_ei));
}

void CudaExpectedImprovementEvaluator::SetupGPU(int devID) {
  OL_CUDA_ERROR_THROW(CudaSetDevice(devID));
}

CudaExpectedImprovementEvaluator::CudaExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in,
                                                                   int num_mc_in, double best_so_far, int devID_in)
      : dim_(gaussian_process_in.dim()),
        num_mc_(num_mc_in),
        best_so_far_(best_so_far),
        gaussian_process_(&gaussian_process_in) {
    SetupGPU(devID_in);
  }

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
  cudaDeviceReset();
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                                           double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled,
                                                           int num_to_sample_in, int num_being_sampled_in,
                                                           bool configure_for_gradients,
                                                           UniformRandomGenerator * uniform_rng_in)
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
      gpu_grad_mu(dim * num_derivatives),
      gpu_chol_var(Square(num_union)),
      gpu_grad_chol_var(dim * Square(num_union) * num_derivatives),
      gpu_ei_storage(kEINumThreads * kEINumBlocks),
      gpu_grad_ei_storage(kGradEINumThreads * kGradEINumBlocks * dim * num_derivatives),
      gpu_random_number_ei(0),
      gpu_random_number_grad_ei(0),
      random_number_ei(0),
      random_number_grad_ei(0) {
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                                           double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled,
                                                           int num_to_sample_in, int num_being_sampled_in,
                                                           bool configure_for_gradients,
                                                           UniformRandomGenerator * uniform_rng_in,
                                                           bool configure_for_test_in)
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
      gpu_grad_mu(dim * num_derivatives),
      gpu_chol_var(Square(num_union)),
      gpu_grad_chol_var(dim * Square(num_union) * num_derivatives),
      gpu_ei_storage(kEINumThreads * kEINumBlocks),
      gpu_grad_ei_storage(kGradEINumThreads * kGradEINumBlocks * dim * num_derivatives),
      gpu_random_number_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kEINumThreads, kEINumBlocks, num_union) : 0),
      gpu_random_number_grad_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kGradEINumThreads, kGradEINumBlocks, num_union) : 0),
      random_number_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kEINumThreads, kEINumBlocks, num_union) : 0),
      random_number_grad_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kGradEINumThreads, kGradEINumBlocks, num_union) : 0) {
}

CudaExpectedImprovementState::CudaExpectedImprovementState(CudaExpectedImprovementState&& OL_UNUSED(other)) = default;

std::vector<double> CudaExpectedImprovementState::BuildUnionOfPoints(double const * restrict points_to_sample,
                                                                     double const * restrict points_being_sampled,
                                                                     int num_to_sample, int num_being_sampled, int dim) noexcept {
  std::vector<double> union_of_points(dim*(num_to_sample + num_being_sampled));
  std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.data());
  std::copy(points_being_sampled, points_being_sampled + dim*num_being_sampled, union_of_points.data() + dim*num_to_sample);
  return union_of_points;
}

int CudaExpectedImprovementState::GetVectorSize(int num_mc_itr, int num_threads, int num_blocks, int num_points) noexcept {
  return ((static_cast<int>(num_mc_itr / (num_threads * num_blocks)) + 1) * (num_threads * num_blocks) * num_points);
}

void CudaExpectedImprovementState::SetCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) {
  // update points_to_sample in union_of_points
  std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

  // evaluate derived quantities for the GP
  points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives);
}

void CudaExpectedImprovementState::SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) {
  // update quantities derived from points_to_sample
  SetCurrentPoint(ei_evaluator, points_to_sample);
}

/*!\rst
  This function is same as ``EvaluateEIAtPointList`` in ``gpp_math.cpp``, except that it is
  specifically used for GPU functions. Refer to ``gpp_math.cpp`` for detailed documentation.
\endrst*/
void CudaEvaluateEIAtPointList(const GaussianProcess& gaussian_process, const ThreadSchedule& thread_schedule,
                               double const * restrict initial_guesses, double const * restrict points_being_sampled,
                               int num_multistarts, int num_to_sample, int num_being_sampled, double best_so_far,
                               int max_int_steps, int which_gpu, bool * restrict found_flag,
                               UniformRandomGenerator * uniform_rng, double * restrict function_values,
                               double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  using DomainType = DummyDomain;
  DomainType dummy_domain;
  bool configure_for_gradients = false;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    EvaluateEIAtPointList(gaussian_process, thread_schedule, initial_guesses, points_being_sampled, num_multistarts,
                          num_to_sample, num_being_sampled, best_so_far, max_int_steps, found_flag, nullptr, function_values, best_next_point);
  } else {
    CudaExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far, which_gpu);

    std::vector<typename CudaExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, points_being_sampled, num_to_sample,
                                  num_being_sampled, thread_schedule.max_num_threads,
                                  configure_for_gradients, uniform_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, initial_guesses);

    NullOptimizer<CudaExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<CudaExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<CudaExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, dummy_domain,
                                            thread_schedule, initial_guesses, num_multistarts,
                                            ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  This function is same as ``ComputeOptimalPointsToSample`` in ``gpp_math.cpp``, except that it is
  specifically used for GPU functions. Refer to ``gpp_math.cpp`` for detailed documentation.
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
                                      double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }

  std::vector<double> next_points_to_sample(gaussian_process.dim()*num_to_sample);

  bool found_flag_local = false;
  if (lhc_search_only == false) {
    CudaComputeOptimalPointsToSampleWithRandomStarts(gaussian_process, optimizer_parameters,
                                                     domain, thread_schedule, points_being_sampled,
                                                     num_to_sample, num_being_sampled,
                                                     best_so_far, max_int_steps, which_gpu,
                                                     &found_flag_local, uniform_generator,
                                                     next_points_to_sample.data());
  }

  // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
  if (found_flag_local == false || lhc_search_only == true) {
    if (unlikely(lhc_search_only == false)) {
      OL_WARNING_PRINTF("WARNING: %d,%d-EI opt DID NOT CONVERGE\n", num_to_sample, num_being_sampled);
      OL_WARNING_PRINTF("Attempting latin hypercube search\n");
    }

    if (num_lhc_samples > 0) {
      // Note: using a schedule different than "static" may lead to flakiness in monte-carlo EI optimization tests.
      // Besides, this is the fastest setting.
      ThreadSchedule thread_schedule_naive_search(thread_schedule);
      thread_schedule_naive_search.schedule = omp_sched_static;
      CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, domain,
                                                              thread_schedule_naive_search,
                                                              points_being_sampled,
                                                              num_lhc_samples, num_to_sample,
                                                              num_being_sampled, best_so_far,
                                                              max_int_steps, which_gpu,
                                                              &found_flag_local, uniform_generator,
                                                              next_points_to_sample.data());

      // if latin hypercube 'dumb' search failed
      if (unlikely(found_flag_local == false)) {
        OL_ERROR_PRINTF("ERROR: %d,%d-EI latin hypercube search FAILED on\n", num_to_sample, num_being_sampled);
      }
    } else {
      OL_WARNING_PRINTF("num_lhc_samples <= 0. Skipping latin hypercube search\n");
    }
  }

  // set outputs
  *found_flag = found_flag_local;
  std::copy(next_points_to_sample.begin(), next_points_to_sample.end(), best_points_to_sample);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, int which_gpu, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
template void CudaComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps,
    bool lhc_search_only, int num_lhc_samples, int which_gpu, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
#endif  // OL_GPU_ENABLED

}  // end namespace optimal_learning
