/*!
  \file gpp_expected_improvement_gpu_test.cpp
  \rst
  Routines to test the functions in gpp_expected_improvement_gpu.cpp.

  The tests verify ExpectedImprovementGPUEvaluator from gpp_expected_improvement_gpu.cpp.

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

  2. Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable

  3. GPU EI vs CPU EI: both use monte-carlo version and run consistency check on various random sample points
\endrst*/

#include "gpp_expected_improvement_gpu_test.hpp"

#include <cstdio>

#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_math_test.hpp"
#include <ctime>

namespace optimal_learning {
    // PingCudaExpectedImprovement function def begins
    PingCudaExpectedImprovement::PingCudaExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled, int num_sampled, int num_mc_iter) 
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_being_sampled_(num_being_sampled),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        points_being_sampled_(points_being_sampled, points_being_sampled + num_being_sampled_*dim_),
        grad_EI_(num_to_sample_*dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, num_mc_iter, best_so_far) {
  }

    void PingCudaExpectedImprovement::GetInputSizes(int * num_rows, int * num_cols) const noexcept {
      *num_rows = dim_;
      *num_cols = num_to_sample_;
    }

    int PingCudaExpectedImprovement::GetGradientsSize() const noexcept {
      return dim_*GetOutputSize();
    }

    int PingCudaExpectedImprovement::GetOutputSize() const noexcept {
      return num_to_sample_;
    }

    void PingCudaExpectedImprovement::EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept {
      if (gradients_already_computed_ == true) {
        OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
      }
      gradients_already_computed_ = true;

      NormalRNG normal_rng(3141);
      bool configure_for_gradients = true;
      CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &normal_rng);
      ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

      if (gradients != nullptr) {
        std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
      }
    }

    double PingCudaExpectedImprovement::GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const {
      if (gradients_already_computed_ == false) {
        OL_THROW_EXCEPTION(RuntimeException, "PingExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
      }

      return grad_EI_[column_index*dim_ + row_index];
    }

    void PingCudaExpectedImprovement::EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept {
      NormalRNG normal_rng(3141);
      bool configure_for_gradients = false;
      CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &normal_rng);
      *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
    }
#ifdef OL_GPU_ENABLED
    /*!\rst
      Generates a set of 50 random test cases for expected improvement with only one potential sample.
      The general EI (which uses MC integration) is evaluated to reasonably high accuracy (while not taking too long to run)
      and compared against the analytic formula version for consistency.  The gradients (spatial) of EI are also checked.

      \return
        number of cases where analytic and monte-carlo EI do not match
    \endrst*/
    int RunCudaEIConsistencyTests() {
      int total_errors = 0;

      const int num_mc_iter = 1000000;
      const int dim = 3;
      const int num_being_sampled = 0;
      const int num_to_sample = 1;
      const int num_sampled = 7;

      double alpha = 2.80723;
      // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
      double best_so_far = 10.0;

        UniformRandomGenerator uniform_generator(31278);
        boost::uniform_real<double> uniform_double(0.5, 2.5);

        MockExpectedImprovementEnvironment EI_environment;

        std::vector<double> lengths(dim);
        std::vector<double> grad_EI_general(dim);
        std::vector<double> grad_EI_one_potential_sample(dim);
        double EI_general;
        double EI_one_potential_sample;

        for (int i = 0; i < 40; ++i) {
          EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
          for (int j = 0; j < dim; ++j) {
            lengths[j] = uniform_double(uniform_generator.engine);
          }
          PingOnePotentialSampleExpectedImprovement EI_one_potential_sample_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
          EI_one_potential_sample_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_one_potential_sample.data());
          EI_one_potential_sample_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_one_potential_sample);

          PingCudaExpectedImprovement EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
          EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_general.data());
          EI_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_general);

          int ei_errors_this_iteration = 0;
          if (!CheckDoubleWithinRelative(EI_general, EI_one_potential_sample, 5.0e-3)) {
            ++ei_errors_this_iteration;
          }
          if (ei_errors_this_iteration != 0) {
            OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
          }
          total_errors += ei_errors_this_iteration;

          int grad_ei_errors_this_iteration = 0;
          for (int j = 0; j < dim; ++j) {
            if (!CheckDoubleWithinRelative(grad_EI_general[j], grad_EI_one_potential_sample[j], 4.5e-3)) {
              ++grad_ei_errors_this_iteration;
            }
          }

          if (grad_ei_errors_this_iteration != 0) {
            OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
          }
          total_errors += grad_ei_errors_this_iteration;
        }

        if (total_errors != 0) {
          OL_PARTIAL_FAILURE_PRINTF("comparing MC EI to analytic EI failed with %d total_errors\n\n", total_errors);
        } else {
          OL_PARTIAL_SUCCESS_PRINTF("comparing MC EI to analytic EI passed\n");
        }

        return total_errors;
    }

    /*!\rst
      Generates a set of 10 random test cases for expected improvement with only one potential sample.
      The general EI (which uses MC integration) is evaluated to reasonably high accuracy (while not taking too long to run)
      and compared against the analytic formula version for consistency.  The gradients (spatial) of EI are also checked.

      \return
        number of cases where analytic and monte-carlo EI do not match
    \endrst*/
    int RunCudaEIvsCpuEI() {
      int total_errors = 0;

      const int num_mc_iter = 20000000;
      const int dim = 3;
      const int num_being_sampled = 4;
      const int num_to_sample = 4;
      const int num_sampled = 20;

      double alpha = 2.80723;
      // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
      double best_so_far = 10.0;

        UniformRandomGenerator uniform_generator(31278);
        boost::uniform_real<double> uniform_double(0.5, 2.5);

        MockExpectedImprovementEnvironment EI_environment;

        std::vector<double> lengths(dim);
        std::vector<double> grad_EI_cpu(dim*num_to_sample);
        std::vector<double> grad_EI_gpu(dim*num_to_sample);
        double EI_cpu;
        double EI_gpu;

        for (int i = 0; i < 10; ++i) {
          EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
          for (int j = 0; j < dim; ++j) {
            lengths[j] = uniform_double(uniform_generator.engine);
          }

          PingExpectedImprovement cpu_EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
          cpu_EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_cpu.data());
          cpu_EI_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_cpu);

          PingCudaExpectedImprovement gpu_EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
          gpu_EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_gpu.data());
          gpu_EI_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_gpu);

          int ei_errors_this_iteration = 0;
          if (!CheckDoubleWithinRelativeWithThreshold(EI_cpu, EI_gpu, 5.0e-4, 1.0e-6)) {
            ++ei_errors_this_iteration;
          }
          if (ei_errors_this_iteration != 0) {
            OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
          }
          total_errors += ei_errors_this_iteration;

          int grad_ei_errors_this_iteration = 0;
          for (int j = 0; j < dim*num_to_sample; ++j) {
            if (!CheckDoubleWithinRelativeWithThreshold(grad_EI_cpu[j], grad_EI_gpu[j], 2.0e-2, 1.0e-3)) {
              ++grad_ei_errors_this_iteration;
            }
          }

          if (grad_ei_errors_this_iteration != 0) {
            OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
          }
          total_errors += grad_ei_errors_this_iteration;
        }

        if (total_errors != 0) {
          OL_PARTIAL_FAILURE_PRINTF("comparing cpu EI to gpu EI failed with %d total_errors\n\n", total_errors);
        } else {
          OL_PARTIAL_SUCCESS_PRINTF("comparing cpu EI to gpu EI passed\n");
        }

        return total_errors;
    }

    void SpeedComparison() {
      const int num_mc_iter = 10000000;
      const int dim = 3;
      const int num_being_sampled = 8;
      const int num_to_sample = 4;
      const int num_sampled = 20;

      double alpha = 2.80723;
      // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
      double best_so_far = 10.0;

        UniformRandomGenerator uniform_generator(31278);
        boost::uniform_real<double> uniform_double(0.5, 2.5);

        MockExpectedImprovementEnvironment EI_environment;

        std::vector<double> lengths(dim);
        std::vector<double> grad_EI_cpu(dim*num_to_sample);
        std::vector<double> grad_EI_gpu(dim*num_to_sample);
        double EI_cpu;
        double EI_gpu;

        EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
        for (int j = 0; j < dim; ++j) {
            lengths[j] = uniform_double(uniform_generator.engine);
        }
        std::vector<double> noise_variance_(num_sampled, 0.0);
        SquareExponential sqexp_covariance_(dim, alpha, lengths);
        GaussianProcess gaussian_process_(sqexp_covariance_, EI_environment.points_sampled(), EI_environment.points_sampled_value(), noise_variance_.data(), EI_environment.dim, EI_environment.num_sampled);
        bool configure_for_gradients = true;
        // gpu computation
        int device_no = 0;
        NormalRNG gpu_normal_rng(3141);
        CudaExpectedImprovementEvaluator gpu_ei_evaluator(gaussian_process_, num_mc_iter, best_so_far, device_no);
        CudaExpectedImprovementEvaluator::StateType gpu_ei_state(gpu_ei_evaluator, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), EI_environment.num_to_sample, EI_environment.num_being_sampled, configure_for_gradients, &gpu_normal_rng);
        std::clock_t start = std::clock();
        gpu_ei_evaluator.ComputeGradExpectedImprovement(&gpu_ei_state, grad_EI_gpu.data());
        double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        printf("gpu grad_EI time duration is: %f\n", duration);

        start = std::clock();
        EI_gpu = gpu_ei_evaluator.ComputeExpectedImprovement(&gpu_ei_state);
        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        printf("gpu EI time duration is: %f\n", duration);

        // cpu computation
        NormalRNG cpu_normal_rng(3141);
        ExpectedImprovementEvaluator cpu_ei_evaluator(gaussian_process_, num_mc_iter, best_so_far);
        ExpectedImprovementEvaluator::StateType cpu_ei_state(cpu_ei_evaluator, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), EI_environment.num_to_sample, EI_environment.num_being_sampled, configure_for_gradients, &cpu_normal_rng);
        start = std::clock();
        cpu_ei_evaluator.ComputeGradExpectedImprovement(&cpu_ei_state, grad_EI_cpu.data());
        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        printf("cpu grad_EI time duration is: %f\n", duration);

        start = std::clock();
        EI_cpu = cpu_ei_evaluator.ComputeExpectedImprovement(&cpu_ei_state);
        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        printf("cpu EI time duration is: %f\n", duration);
    }
#else
    int RunCudaEIConsistencyTests() {
        printf("no gpu component is enabled, this test did not run.\n");
        return 0;
    }
    int RunCudaEIvsCpuEI() {
        printf("no gpu component is enabled, this test did not run.\n");
        return 0;
    }
    void SpeedComparison() {
        printf("no gpu component is enabled, this test did not run.\n");
    }
#endif
}

