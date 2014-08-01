/*!
  \file gpp_heuristic_expected_improvement_optimization.cpp
  \rst
  This file contains defintions of expensive ObjectiveEstimationPolicyInterface::ComputeEstimate() functions as well
  as the function ComputeHeuristicPointsToSample() which uses these policies to heuristically optimize the
  q-EI problem. The idea behind the latter is to make explicit guesses about the behavior of the underlying objective
  function (that the GaussianProcess is modeling), which is cheap, instead of using the GP's more powerful notion
  of the distribution of possible objective function behaviors. That is, instead of taking expectations on the
  distribution of objective function behaviors, we simply pick one (through the EstimationPolicy).

  Readers should review the header docs for gpp_math.hpp/cpp first to understand Gaussian Processes and Expected
  Improvement.
\endrst*/

#include "gpp_heuristic_expected_improvement_optimization.hpp"

#include <cmath>

#include <memory>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

struct NormalRNG;

/*!\rst
  .. NOTE:: Depending on the use-case, performance could improve if the GaussianProcess were stored as a class member
      alongside a matching PointsToSampleState. That said, doing so introduces new issues in maintaining consistency.
      It is not a performance concern right now (this function is called infrequently).
\endrst*/
FunctionValue KrigingBelieverEstimationPolicy::ComputeEstimate(const GaussianProcess& gaussian_process, double const * restrict point, int OL_UNUSED(iteration)) const {
  const int num_points = 1;
  const int num_derivatives = 0;
  PointsToSampleState gaussian_process_state(gaussian_process, point, num_points, num_derivatives);

  double kriging_function_value;
  gaussian_process.ComputeMeanOfPoints(gaussian_process_state, &kriging_function_value);
  if (std_deviation_coef_ != 0.0) {
    // Only compute variance (expensive) if we are going to use it.
    double gp_variance;
    gaussian_process.ComputeVarianceOfPoints(&gaussian_process_state, &gp_variance);
    kriging_function_value += std_deviation_coef_ * std::sqrt(gp_variance);
  }

  return FunctionValue(kriging_function_value, kriging_noise_variance_);
}

/*!\rst
  This implements a generic tool for heuristically solving the q-EI problem using methods like "Constant Liar" or
  "Kriging Believer" described in Ginsbourger 2008. In a loop, we solve 1-EI (with resulting optima "point"), then ask
  a heuristic EstimationPolicy to guess the objective function value at "point" (in lieu of sampling the real objective by
  say, running an [expensive] experiment).

  As such, this method is really a fairly loose wrapper around ComputeOptimalPointsToSampleWithRandomStarts() configured
  to optimize 1-EI.

  Solving q-EI optimally is expensive since this requires monte-carlo evaluation of EI and its gradient. This method
  is much cheaper: 1-EI allows analytic computation of EI and its gradient and is fairly easily optimized. So this
  heuristic optimizer is cheaper but potentially highly inaccurate, providing no guarantees on the quality of the
  best_points_to_sample output.
\endrst*/
template <typename DomainType>
void ComputeHeuristicPointsToSample(const GaussianProcess& gaussian_process,
                                    const GradientDescentParameters& optimizer_parameters,
                                    const DomainType& domain,
                                    const ObjectiveEstimationPolicyInterface& estimation_policy,
                                    const ThreadSchedule& thread_schedule,
                                    double best_so_far, bool lhc_search_only, int num_lhc_samples,
                                    int num_to_sample, bool * restrict found_flag,
                                    UniformRandomGenerator * uniform_generator,
                                    double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }
  const int dim = gaussian_process.dim();
  // For speed, we stick to the analytic EI routines: so we only consider q,0-EI (q-EI), not the more general q,p-EI problem;
  // and we estimate its solution as a sequence of 1,0-EI problems.
  const int num_to_sample_per_iteration = 1;
  const int num_being_sampled = 0;
  const int max_int_steps = 0;
  double * const points_being_sampled = nullptr;
  NormalRNG * const normal_rng = nullptr;

  // Cannot/Should not modify the input gaussian_process (the GP with the estimated objective values is of little use;
  // the caller at most wants to do other optimization tasks on the GP with only prior data), so Clone() it first and
  // work with the clone.
  std::unique_ptr<GaussianProcess> gaussian_process_local(gaussian_process.Clone());

  bool found_flag_overall = true;
  for (int i = 0; i < num_to_sample; ++i) {
    bool found_flag_local = false;
    if (likely(lhc_search_only == false)) {
      ComputeOptimalPointsToSampleWithRandomStarts(*gaussian_process_local, optimizer_parameters,
                                                   domain, thread_schedule, points_being_sampled,
                                                   num_to_sample_per_iteration, num_being_sampled,
                                                   best_so_far, max_int_steps, &found_flag_local,
                                                   uniform_generator, normal_rng, best_points_to_sample);
    }
    // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
    if (unlikely(found_flag_local == false || lhc_search_only == true)) {
      if (unlikely(lhc_search_only == false)) {
        OL_WARNING_PRINTF("WARNING: Heuristic EI opt DID NOT CONVERGE on iteration %d of %d\n", i, num_to_sample);
        OL_WARNING_PRINTF("Attempting latin hypercube search\n");
      }

      if (num_lhc_samples > 0) {
        // This is the fastest setting.
        ThreadSchedule thread_schedule_naive_search(thread_schedule);
        thread_schedule_naive_search.schedule = omp_sched_static;
        ComputeOptimalPointsToSampleViaLatinHypercubeSearch(*gaussian_process_local, domain,
                                                            thread_schedule_naive_search,
                                                            points_being_sampled, num_lhc_samples,
                                                            num_to_sample_per_iteration,
                                                            num_being_sampled, best_so_far,
                                                            max_int_steps,
                                                            &found_flag_local, uniform_generator,
                                                            normal_rng, best_points_to_sample);

        // if latin hypercube 'dumb' search failed
        if (unlikely(found_flag_local == false)) {
          OL_ERROR_PRINTF("ERROR: Heuristic EI latin hypercube search FAILED on iteration %d of %d\n", i, num_to_sample);
          *found_flag = false;
          return;
        }
      } else {
        OL_WARNING_PRINTF("num_lhc_samples <= 0. Skipping latin hypercube search\n");
      }
    }

    FunctionValue function_estimate = estimation_policy.ComputeEstimate(*gaussian_process_local,
                                                                        best_points_to_sample, i);
    gaussian_process_local->AddPointsToGP(best_points_to_sample, &function_estimate.function_value,
                                         &function_estimate.noise_variance, 1);

    found_flag_overall &= found_flag_local;
    best_points_to_sample += dim;
  }

  *found_flag = found_flag_overall;
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void ComputeHeuristicPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy,
    const ThreadSchedule& thread_schedule, double best_so_far, bool lhc_search_only,
    int num_lhc_samples, int num_to_sample, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
template void ComputeHeuristicPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy,
    const ThreadSchedule& thread_schedule, double best_so_far, bool lhc_search_only,
    int num_lhc_samples, int num_to_sample, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

}  // end namespace optimal_learning
