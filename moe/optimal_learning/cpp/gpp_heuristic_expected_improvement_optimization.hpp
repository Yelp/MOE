/*!
  \file gpp_heuristic_expected_improvement_optimization.hpp
  \rst
  1. FILE OVERVIEW
  2. CODE DESIGN/LAYOUT OVERVIEW:

     a. class FunctionValue
     b. class ObjectiveEstimationPolicyInterface

        i. ConstantLiarEstimationPolicy
        ii. KrigingBelieverEstimationPolicy

     c. function ComputeHeuristicPointsToSample()

  **1 FILE OVERVIEW**

  Readers should review the header docs for gpp_math.hpp first to understand Gaussian Processes and Expected
  Improvement; readers should additionally check gpp_math.cpp file docs for further details.

  This file declares classes and functions supporting "heuristic" Expected Improvement optimization to solve the
  q,0-EI problem. In gpp_math.hpp, methods like ComputeOptimalPointsToSample() solve the q,p-EI problem
  using only intrinsic properties of a GaussianProcess. Here, we make additional assumptions about the underlying
  objective function's behavior, trading accuracy for increased speed.

  Together, the ConstantLiar and KrigingBeliever estimation policies with ComputeHeuristicPointsToSample()
  implement the heuristics discussed in Ginsbourger 2008. Now we'll provide a brief overview of these components;
  see the class and function docs for more details.

  **2 CODE DESIGN/LAYOUT OVERVIEW**

  Currently, we have:

  **2a FunctionValue**

  A simple container class for holding the pair (function_value, noise_variance), representing a measured or
  estimated objective function value and the associated noise variance.

  **2b ObjectiveEstimationPolicyInterface**

  A simple interface for computing objective function estimates. This supports a single function, ComputeEstimate(),
  that estimates the objective function evaluated at a point. It additionally has access to the GaussianProcess
  and an iteration counter.

  **2b, i ConstantLiarEstimationPolicy**

  The simplest estimation policy, "Constant Liar" always returns the same objective function estimate, no matter what.

  **2b, ii KrigingBelieverEstimationPolicy**

  Kriging Believer uses some information from the GaussianProcess to produce its estimates. In the basic form
  (as used in Ginsbourger 2008), Kriging returns the GP Mean at the evaluation point. We also allow shifting
  by some scaling of the GP std deviation.

  **2c Finally, we discuss performing heuristic EI optimization via ComputeHeuristicPointsToSample()**

  As with the EI optimizers in gpp_math, this function is templated on domain. This function is responsible for
  actually performing the heuristic optimization. It uses ComputeOptimalPointsToSampleWithRandomStarts()
  (from gpp_math.hpp) to do this. This function estimates the solution to q-EI using a sequence of q solves
  of the 1-EI problem--cheaper, but potentially/probably inaccurate.

  Lastly, note that this template function is explicitly instantiated. There are limited domain choices, and this
  lets us hide all the implementation details in the .cpp file (e.g., see the long list of forward declarations).
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

class TensorProductDomain;
class SimplexIntersectTensorProductDomain;
struct UniformRandomGenerator;
class GaussianProcess;
struct GradientDescentParameters;
struct ThreadSchedule;

/*!\rst
  Enumerating estimation policies for convenience. Useful for dispatching tests.
\endrst*/
enum class EstimationPolicyTypes {
  //! ConstantLiarEstimationPolicy
  kConstantLiar = 0,
  //! KrigingBelieverEstimationPolicy
  kKrigingBeliever = 1,
};

/*!\rst
  Container (POD) to represent the notion of a measured function value with some [Gaussian] uncertainty.
  That is, ``N(\mu, \sigma)``, where ``\mu = function_value`` and ``\sigma = \sqrt{noise_variance}``.
\endrst*/
struct FunctionValue {
  /*!\rst
    Explicitly defaulted default constructor.
    Defining a custom ctor (below) disables the default ctor, so we explicitly default it.
    This is needed to maintain POD-ness.
  \endrst*/
  FunctionValue() = default;

  /*!\rst
    Constructs a FunctionValue object with the specified function_value and 0 noise.

    \param
      :function_value_in: the function_value to hold
  \endrst*/
  explicit FunctionValue(double function_value_in) : FunctionValue(function_value_in, 0.0) {
  }

  /*!\rst
    Constructs a FunctionValue object with the specified function_value noise_variance.

    \param
      :function_value_in: the function_value to hold
      :noise_variance_in: the noise_variance to hold
  \endrst*/
  FunctionValue(double function_value_in, double noise_variance_in) : function_value(function_value_in), noise_variance(noise_variance_in) {
  }

  //! the measured function value being represented
  double function_value;
  //! the uncertainty (variance) in the measurement of the function value
  double noise_variance;
};

/*!\rst
  At the moment, ComputeEstimatedSetOfPointsToSample() is the sole consumer of this class. Some of the documentation
  here will discuss ObjectiveEstimationPolicyInterface specifically in that light.

  He also points out that larger lie values will lead methods like ComputeEstimatedSetOfPointsToSample() to
  be more explorative and vice versa.
\endrst*/
class ObjectiveEstimationPolicyInterface {
 public:
  virtual ~ObjectiveEstimationPolicyInterface() = default;

  /*!\rst
    The estimate is often computed repeatedly (e.g., see ComputeEstimatedSetOfPointsToSample()); we include
    the number of previous calls in the input "iteration." This may be useful if users want to implement an
    analogue of "Constant Liar" using a fixed distribution of lies, make random draws, etc.

    Let dim = gaussian_process.dim()

    \param
      :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      :point[dim]: the point at which to compute the estimate
      :iteration: the number of previous calls to ComputeEstimate()
  \endrst*/
  virtual FunctionValue ComputeEstimate(const GaussianProcess& gaussian_process, double const * restrict point, int iteration) const OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT = 0;
};

/*!\rst
  The "Constant Liar" objective function estimation policy is the simplest: it always returns the same value
  (Ginsbourger 2008). We call this the "lie". This object also allows users to associate a noise variance to
  the lie value.

  In Ginsbourger's work, the most common lie values have been the min and max of all previously observed objective
  function values; i.e., min, max of ``GP.points_sampled_value``. The mean has also been considered.

  He also points out that larger lie values (e.g., max of prior measurements) will lead methods like
  ComputeEstimatedSetOfPointsToSample() to be more explorative and vice versa.

  .. Note:: These comments were copied into constant_liar_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py.
\endrst*/
class ConstantLiarEstimationPolicy final : public ObjectiveEstimationPolicyInterface {
 public:
  /*!\rst
    Constructs a ConstantLiarEstimationPolicy object with the specified lie_value and 0 noise.

    \param
      :lie_value: the "constant lie" that this estimator should return
  \endrst*/
  explicit ConstantLiarEstimationPolicy(double lie_value) : ConstantLiarEstimationPolicy(lie_value, 0.0) {
  }

  /*!\rst
    Constructs a ConstantLiarEstimationPolicy object with the specified lie_value and lie_noise_variance.

    \param
      :lie_value: the "constant lie" that this estimator should return
      :lie_noise_variance: the ``noise_variance`` to associate to the ``lie_value`` (MUST be ``>= 0.0``)
  \endrst*/
  ConstantLiarEstimationPolicy(double lie_value, double lie_noise_variance) : lie_value_(lie_value), lie_noise_variance_(lie_noise_variance) {
  }

  virtual FunctionValue ComputeEstimate(const GaussianProcess& OL_UNUSED(gaussian_process), double const * restrict OL_UNUSED(point), int OL_UNUSED(iteration)) const OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return FunctionValue(lie_value_, lie_noise_variance_);
  }

  ConstantLiarEstimationPolicy() = delete;

 private:
  //! the constant function value estimate this object should return
  double lie_value_;
  //! the noise variance to associate to the lie_value
  double lie_noise_variance_;
};

/*!\rst
  The "Kriging Believer" objective function estimation policy uses the Gaussian Process (i.e., the prior)
  to produce objective function estimates. The simplest method is to trust the GP completely:
  ``estimate = GP.mean(point)``
  This follows the usage in Ginsbourger 2008. Users may also want the estimate to depend on the GP variance
  at the evaluation point, so that the estimate reflects how confident the GP is in the prediction. Users may
  also specify std_devation_ceof:
  ``estimate = GP.mean(point) + std_deviation_coef * GP.variance(point)``
  Note that the coefficient is signed, and analogously to ConstantLiar, larger positive values are more
  explorative and larger negative values are more exploitive.

  This object also allows users to associate a noise variance to the lie value.

  .. Note:: These comments were copied into kriging_believer_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py.
\endrst*/
class KrigingBelieverEstimationPolicy final : public ObjectiveEstimationPolicyInterface {
 public:
  /*!\rst
    Constructs a KrigingBelieverEstimationPolicy object whose estimates will only depend on the mean with 0 noise.
  \endrst*/
  KrigingBelieverEstimationPolicy() : KrigingBelieverEstimationPolicy(0.0) {
  }

  /*!\rst
    Constructs a KrigingBelieverEstimationPolicy object with the specified std_deviation_coef and 0 noise.

    \param
      :std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
  \endrst*/
  explicit KrigingBelieverEstimationPolicy(double std_deviation_coef) : KrigingBelieverEstimationPolicy(std_deviation_coef, 0.0) {
  }

  /*!\rst
    Constructs a KrigingBelieverEstimationPolicy object with the specified std_deviation_coef and 0 noise.

    \param
      :std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
      :kriging_noise_variance: the ``noise_variance`` to associate to each function value estimate (MUST be ``>= 0.0``)
  \endrst*/
  KrigingBelieverEstimationPolicy(double std_deviation_coef, double kriging_noise_variance) : std_deviation_coef_(std_deviation_coef), kriging_noise_variance_(kriging_noise_variance) {
  }

  virtual FunctionValue ComputeEstimate(const GaussianProcess& gaussian_process, double const * restrict point, int OL_UNUSED(iteration)) const OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

 private:
  //! the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
  double std_deviation_coef_;
  //! the noise variance to associate to each KrigingBeliever estimate
  double kriging_noise_variance_;
};

/*!\rst
  This function computes a heuristic approximation to the result of ComputeOptimalPointsToSample() with 0 ongoing
  experiments (``points_being_sampled``). Consider this as an alternative when ComputeOptimalPointsToSample() is too expensive.

  It heuristically solves the q,0-EI optimization problem. As a reminder, that problem is finding the set of q points
  that maximizes the Expected Improvement (saved in the output, best_points_to_sample). Solving for q points simultaneously
  usually requires monte-carlo iteration and is expensive. The heuristic here solves q-EI as a sequence of 1-EI problems.
  We solve 1-EI, and then we *ASSUME* an objective function value at the resulting optima. This process is repeated q times.
  It is perhaps more clear in pseudocode::

    points_being_sampled = {}  // This stays empty! We are only working with 1,0-EI solves
    for i = 0:num_to_sample-1 {
      // First, solve the 1,0-EI problem\*
      new_point = ComputeOptimalPointsToSampleWithRandomStarts(gaussian_process, points_being_sampled, other_parameters)
      // *Estimate* the objective function value at new_point
      new_function_value = ESTIMATED_OBJECTIVE_FUNCTION_VALUE(new_point, other_args)
      new_function_value_noise = ESTIMATED_NOISE_VARIANCE(new_point, other_args)
      // Write the estimated objective values to the GP as *truth*
      gaussian_process.AddPoint(new_point, new_function_value, new_function_value_noise)
      optimal_points_to_sample.append(new_point)
    }

  \*Recall: each call to ComputeOptimalPointsToSampleWithRandomStarts() (gpp_math.hpp) kicks off a round of MGD optimization of 1-EI.

  Note that ideally the estimated objective function value (and noise) would be measured from the real-world (e.g.,
  by running an experiment). Then this algorithm would be optimal. However, the estimate probably is not accurately
  representating of the true objective.

  The estimation is handled through the "estimation_policy" input. Passing a ConstantLiarEstimationPolicy or
  KrigingBelieverEstimationPolicy object to this function will produce the "Constant Liar" and "Kriging Believer"
  heuristics described in Ginsbourger 2008. The interface for estimation_policy is generic so users may specify
  other estimators as well.

  Contrast this approach with ComputeOptimalPointsToSample() (gpp_math.hpp) which solves all outputs of the q,0-EI
  problem simultaneously instead of one point at a time. That method is more accurate (b/c it
  does not attempt to estimate the behavior of the underlying objective function) but much more expensive (because it
  requires monte-carlo iteration).

  If ``num_to_sample = 1``, this is exactly the same as ComputeOptimalPointsToSample(); i.e.,
  both methods solve the 1-EI optimization problem the same way.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  local optima (i.e., the gradient may be substantially nonzero).

  WARNING: this function fails if any step fails to find improvement! In that case, the best_points output should not be
           read and found_flag will be false.

  .. NOTE:: These comments were copied into _heuristic_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py.

  \param
    :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
      underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling 1-EI optimization (e.g., number
      of iterations, tolerances, learning rate) in each "outer" iteration
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :estimation_policy: the policy to use to produce (heuristic) objective function estimates during q,0-EI optimization
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :best_so_far: value of the best sample so far (must be min(points_sampled_value))
    :lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
    :num_lhc_samples: number of samples to draw if/when doing latin hypercube search
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if best_points_to_sample corresponds to a nonzero EI if sampled simultaneously
    :uniform_generator[1]:UniformRandomGenerator object will have its state changed due to random draws
    :best_points_to_sample[num_to_sample*dim]: point yielding the best EI according to constant liar
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
                                    double * restrict best_points_to_sample);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeHeuristicPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy,
    const ThreadSchedule& thread_schedule, double best_so_far, bool lhc_search_only,
    int num_lhc_samples, int num_to_sample, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);
extern template void ComputeHeuristicPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ObjectiveEstimationPolicyInterface& estimation_policy,
    const ThreadSchedule& thread_schedule, double best_so_far, bool lhc_search_only,
    int num_lhc_samples, int num_to_sample, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, double * restrict best_points_to_sample);

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_

