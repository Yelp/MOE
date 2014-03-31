// gpp_heuristic_expected_improvement_optimization.cpp
/*
  This file contains defintions of expensive ObjectiveEstimationPolicyInterface::ComputeEstimate() functions as well
  as the function ComputeHeuristicSetOfPointsToSample() which uses these policies to heuristically optimize the
  q-EI problem. The idea behind the latter is to make explicit guesses about the behavior of the underlying objective
  function (that the GaussianProcess is modeling), which is cheap, instead of using the GP's more powerful notion
  of the distribution of possible objective function behaviors. That is, instead of taking expectations on the
  distribution of objective function behaviors, we simply pick one (through the EstimationPolicy).

  Readers should review the header docs for gpp_math.hpp/cpp first to understand Gaussian Processes and Expected
  Improvement.
*/

#include "gpp_heuristic_expected_improvement_optimization.hpp"

#include <cmath>

#include <memory>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization_parameters.hpp"

namespace optimal_learning {

/*
  NOTE: Depending on the use-case, performance could improve if the GaussianProcess were stored as a class member
  alongside a matching PointsToSampleState. That said, doing so introduces new issues in maintaining consistency.
  It is not a performance concern right now (this function is called infrequently).
*/
FunctionValue KrigingBelieverEstimationPolicy::ComputeEstimate(const GaussianProcess& gaussian_process, double const * restrict point, int OL_UNUSED(iteration)) const {
  const int num_points = 1;
  const int configure_for_gradients = false;
  PointsToSampleState gaussian_process_state(gaussian_process, point, num_points, configure_for_gradients);

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

/*
  This implements a generic tool for heuristically solving the q-EI problem using methods like "Constant Liar" or
  "Kriging Believer" described in Ginsbourger 2008. In a loop, we solve 1-EI (with resulting optima "point"), then ask
  a heuristic EstimationPolicy to guess the objective function value at "point" (in lieu of sampling the real objective by
  say, running an [expensive] experiment).

  As such, this method is really a fairly loose wrapper around ComputeOptimalPointToSampleWithRandomStarts() configured
  to optimize 1-EI.

  Solving q-EI optimally is expensive since this requires monte-carlo evaluation of EI and its gradient. This method
  is much cheaper: 1-EI allows analytic computation of EI and its gradient and is fairly easily optimized. So this
  heuristic optimizer is cheaper but potentially highly inaccurate, providing no guarantees on the quality of the
  best_points_to_sample output.
*/
template <typename DomainType>
void ComputeHeuristicSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const DomainType& domain, const ObjectiveEstimationPolicyInterface& estimation_policy, double best_so_far, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample) {
  if (unlikely(num_samples_to_generate <= 0)) {
    return;
  }
  const int dim = gaussian_process.dim();
  const int num_to_sample = 0;

  // Cannot/Should not modify the input gaussian_process (the GP with the estimated objective values is of little use;
  // the caller at most wants to do other optimization tasks on the GP with only prior data), so Clone() it first and
  // work with the clone.
  std::unique_ptr<GaussianProcess> gaussian_process_local(gaussian_process.Clone());

  bool found_flag_overall = true;
  for (int i = 0; i < num_samples_to_generate; ++i) {
    bool found_flag_local = false;
    if (likely(lhc_search_only == false)) {
      ComputeOptimalPointToSampleWithRandomStarts(*gaussian_process_local, optimization_parameters, domain, nullptr, num_to_sample, best_so_far, 0, max_num_threads, &found_flag_local, uniform_generator, nullptr, best_points_to_sample);
    }
    // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
    if (unlikely(found_flag_local == false || lhc_search_only == true)) {
      if (unlikely(lhc_search_only == false)) {
        OL_WARNING_PRINTF("WARNING: Constant Liar EI opt DID NOT CONVERGE on iteration %d of %d\n", i, num_samples_to_generate);
        OL_WARNING_PRINTF("Attempting latin hypercube search\n");
      }

      const int max_int_steps = 0;  // always hitting the analytic case
      ComputeOptimalPointToSampleViaLatinHypercubeSearch(*gaussian_process_local, domain, nullptr, num_lhc_samples, num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag_local, uniform_generator, nullptr, best_points_to_sample);

      // if latin hypercube 'dumb' search failed
      if (unlikely(found_flag_local == false)) {
        OL_ERROR_PRINTF("ERROR: Constant Liar EI latin hypercube search FAILED on iteration %d of %d\n", i, num_samples_to_generate);
        *found_flag = false;
        return;
      }
    }

    FunctionValue function_estimate = estimation_policy.ComputeEstimate(*gaussian_process_local, best_points_to_sample, i);
    gaussian_process_local->AddPointToGP(best_points_to_sample, function_estimate.function_value, function_estimate.noise_variance);

    found_flag_overall &= found_flag_local;
    best_points_to_sample += dim;
  }

  *found_flag = found_flag_overall;
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void ComputeHeuristicSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const TensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy, double best_so_far, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
template void ComputeHeuristicSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const SimplexIntersectTensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy, double best_so_far, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

}  // end namespace optimal_learning
