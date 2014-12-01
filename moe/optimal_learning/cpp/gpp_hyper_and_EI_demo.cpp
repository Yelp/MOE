/*!
  \file gpp_hyper_and_EI_demo.cpp
  \rst
  ``moe/optimal_learning/cpp/gpp_hyper_and_EI_demo.cpp``

  This demo combines gpp_hyperparameter_optimization_demo.cpp and gpp_expected_improvement_demo.cpp.  If you have read
  and understood those, then this demo should be very straightforward insofar as it is currently almost a direct copy-paste.

  The purpose here is to give an "end to end" demo of how someone might use MOE/OL to generate new experimental cohorts,
  beginning with a set of known experimental cohorts/objective function values, measurement noise, and knowledge of any
  ongoing experiments.

  The basic layout is:

  1. Set up input data sizes
  2. Generate random hyperparameters
  3. Generate (random) set of sampled point locations, noise variances
  4. Use a randomly constructed (from inputs in steps 1-3) Gaussian Process (generator) to generate imaginary objective function values
  5. Optimize hyperparameters on the constructed function values
  6. Select desired concurrent experiment locations (points_being_sampled)
  7. Construct Gaussian Process (model) to model the training data "world," using the optimized hyperparameters
  8. Optimize Expected Improvement to decide what point we would sample next

     a. Do this once using the optimized hyperparameters
     b. And again using wrong hyperparameters to emulate a human not knowing how to pick (but drawing from a GP with the same state).
        To do this, we build another GP (wrong_hyper) using the wrong hyperparameters but the same training data as the model gp
     c. Compare resulting function values

  Steps 1-4 happen in both other demos.  Step 5 is the heart of gpp_hyperparameter_optimization_demo.cpp and steps 6-7 are
  the heart of gpp_expected_improvement_demo.cpp.

  Please read and understand the file comments for gpp_expected_improvement_demo.cpp (first) and
  gpp_hyperparameter_optimization_demo.cpp (second) before going through this demo.  The comments are a lot sparser here
  than in the aforementioned two files to avoid redundancy.
\endrst*/

#include <cstdio>

#include <algorithm>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

using namespace optimal_learning;  // NOLINT, i'm lazy in this file which has no external linkage anyway
int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  // here we set some configurable parameters
  // feel free to change them (and recompile) as you explore
  // comments next to each parameter will indicate its purpose and domain

  // the "spatial" dimension, aka the number of independent (experiment) parameters
  static const int dim = 3;  // > 0

  // number of points to optimize simultaneously (for simult experiments); "q" in q,p-EI
  static const int num_to_sample = 1;  // >= 1

  // number of concurrent samples running alongside the optimization; "p" in q,p-EI
  static const int num_being_sampled = 2;  // >= 0

  // number of points that we have already sampled; i.e., size of the training set
  static const int num_sampled = 10;  // >= 0

  UniformRandomGenerator uniform_generator(314);  // repeatable results
  // construct with (base_seed, thread_id) to generate a 'random' seed

  // specifies the domain of each independent variable in (min, max) pairs
  std::vector<ClosedInterval> domain_bounds = {
    {-1.5, 2.3},
    {0.1,  3.1},
    {1.7,  2.9}};
  DomainType domain(domain_bounds.data(), dim);

  // now we allocate point sets; ALL POINTS MUST LIE INSIDE THE DOMAIN!
  std::vector<double> points_sampled(num_sampled*dim);

  std::vector<double> points_sampled_value(num_sampled);

  // default to 0 noise
  std::vector<double> noise_variance(num_sampled, 0.0);  // each entry must be >= 0.0

  // covariance selection
  using CovarianceClass = SquareExponential;  // see gpp_covariance.hpp for other options

  // arbitrary hyperparameters used to generate data
  std::vector<double> hyperparameters_original(1 + dim);
  // generate randomly
  boost::uniform_real<double> uniform_double_for_hyperparameter(0.5, 1.5);
  CovarianceClass covariance_original(dim, 1.0, 1.0);
  FillRandomCovarianceHyperparameters(uniform_double_for_hyperparameter, &uniform_generator,
                                      &hyperparameters_original, &covariance_original);

  // Generate data that will be used to build the GP
  // set noise
  std::fill(noise_variance.begin(), noise_variance.end(), 1.0e-1);  // arbitrary choice

  // use latin hypercube sampling to get a reasonable distribution of training point locations
  domain.GenerateUniformPointsInDomain(num_sampled, &uniform_generator, points_sampled.data());

  // build an empty GP: since num_sampled (last arg) is 0, none of the data arrays will be used here
  GaussianProcess gp_generator(covariance_original, points_sampled.data(), points_sampled_value.data(),
                               noise_variance.data(), dim, 0);
  // fill the GP with randomly generated data
  FillRandomGaussianProcess(points_sampled.data(), noise_variance.data(), dim, num_sampled,
                            points_sampled_value.data(), &gp_generator);

  // simulate not knowing the hyperparameters by choosing a new set randomly (start hyperparameter opt from here)
  std::vector<double> hyperparameters_perturbed(covariance_original.GetNumberOfHyperparameters());
  // choose from a small-ish range; for the demo we want hyperparameter opt to converge fairly quickly/robustly
  boost::uniform_real<double> uniform_double_for_perturbed_hyperparameter(4.0, 8.0);
  CovarianceClass covariance_perturbed(dim, 1.0, 1.0);
  FillRandomCovarianceHyperparameters(uniform_double_for_perturbed_hyperparameter, &uniform_generator,
                                      &hyperparameters_perturbed, &covariance_perturbed);

  std::vector<ClosedInterval> hyperparameter_domain_bounds(covariance_original.GetNumberOfHyperparameters(), {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(),
                                                 covariance_original.GetNumberOfHyperparameters());

  // Log Likelihood eval
  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  // log likelihood evaluator object
  LogLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(),
                                           noise_variance.data(), dim, num_sampled);

  // Newton setup
  int total_newton_errors = 0;  // number of newton runs that failed due to singular hessians
  int newton_max_num_steps = 500;  // max number of newton steps
  double gamma_newton = 1.05;  // newton diagonal dominance scale-down factor (see newton docs for details)
  double pre_mult_newton = 1.0e-1;  // newton diagonal dominance scaling factor (see newton docs for details)
  double max_relative_change_newton = 1.0;
  double tolerance_newton = 1.0e-11;
  NewtonParameters newton_parameters(1, newton_max_num_steps, gamma_newton, pre_mult_newton,
                                     max_relative_change_newton, tolerance_newton);

  std::vector<double> new_newton_hyperparameters(covariance_original.GetNumberOfHyperparameters());

  printf(OL_ANSI_COLOR_CYAN "ORIGINAL HYPERPARMETERS:\n" OL_ANSI_COLOR_RESET);
  printf("Original Hyperparameters:\n");
  PrintMatrix(hyperparameters_original.data(), 1, covariance_original.GetNumberOfHyperparameters());

  printf(OL_ANSI_COLOR_CYAN "NEWTON OPTIMIZED HYPERPARAMETERS:\n" OL_ANSI_COLOR_RESET);
  // run newton optimization
  total_newton_errors += NewtonHyperparameterOptimization(log_marginal_eval, covariance_perturbed,
                                                          newton_parameters, hyperparameter_domain,
                                                          new_newton_hyperparameters.data());

  printf("Result of newton:\n");
  PrintMatrix(new_newton_hyperparameters.data(), 1, covariance_original.GetNumberOfHyperparameters());

  if (total_newton_errors > 0) {
    printf("WARNING: %d newton runs exited due to singular Hessian matrices.\n", total_newton_errors);
  }

  // Now optimize EI using the 'best' hyperparameters
  // set gaussian process's hyperparameters to the result of newton optimization
  CovarianceClass covariance_opt(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
  GaussianProcess gp_model(covariance_opt, points_sampled.data(), points_sampled_value.data(),
                           noise_variance.data(), dim, num_sampled);

  // remaining inputs to EI optimization
  // just an arbitrary point set for when num_being_sampled = 2, as in the default setting for this demo
  std::vector<double> points_being_sampled(num_being_sampled*dim);
  if (num_being_sampled == 2) {
    points_being_sampled[0] = 0.3; points_being_sampled[1] = 2.7; points_being_sampled[2] = 2.2;
    points_being_sampled[3] = -0.2; points_being_sampled[4] = 0.6; points_being_sampled[5] = 1.9;
  }

  // multithreading
  int max_num_threads = 2;  // feel free to experiment with different numbers
  ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);

  // set up RNG containers
  int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};  // arbitrarily used digits of pi as seeds
  std::vector<NormalRNG> normal_rng_vec(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    normal_rng_vec[i].SetExplicitSeed(pi_array[i]);  // to get repeatable results
    // call SetRandomizedSeed(base_seed, thread_id) to automatically choose 'random' seeds
  }

  double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());  // this is simply the best function value seen to date

  // gradient descent parameters
  double gamma = 0.1;  // we decrease step size by a factor of 1/(iteration)^gamma
  double pre_mult = 1.0;  // scaling factor
  double max_relative_change_ei = 1.0;
  double tolerance_ei = 1.0e-7;
  int num_multistarts = 10;  // max number of multistarted locations
  int max_num_steps = 500;  // maximum number of GD iterations per restart
  int max_num_restarts = 20;  // number of restarts to run with GD
  int num_steps_averaged = 0;  // number of steps to use in polyak-ruppert averaging
  GradientDescentParameters gd_params(num_multistarts, max_num_steps, max_num_restarts,
                                      num_steps_averaged, gamma,
                                      pre_mult, max_relative_change_ei, tolerance_ei);

  // EI evaluation parameters
  int max_int_steps = 1000;  // number of monte carlo iterations
  std::vector<double> next_point_winner(dim);

  {  // optimize EI using a model with the optimized hyperparameters
    printf(OL_ANSI_COLOR_CYAN "OPTIMIZING EXPECTED IMPROVEMENT... (optimized hyperparameters)\n" OL_ANSI_COLOR_RESET);
    bool found_flag = false;
    ComputeOptimalPointsToSampleWithRandomStarts(gp_model, gd_params, domain, thread_schedule,
                                                 points_being_sampled.data(), num_to_sample,
                                                 num_being_sampled, best_so_far, max_int_steps,
                                                 &found_flag, &uniform_generator, normal_rng_vec.data(),
                                                 next_point_winner.data());
    printf(OL_ANSI_COLOR_CYAN "EI OPTIMIZATION FINISHED (optimized hyperparameters). Success status: %s\n" OL_ANSI_COLOR_RESET, found_flag ? "True" : "False");
    printf("Next best sample point according to EI (opt hyper):\n");
    PrintMatrix(next_point_winner.data(), 1, dim);

    // check what the actual improvement would've been by sampling from our GP and comparing to best_so_far
    // put randomness in a known state
    gp_generator.SetExplicitSeed(31415);
    double function_value = gp_generator.SamplePointFromGP(next_point_winner.data(), 0.0);  // sample w/o noise

    printf(OL_ANSI_COLOR_CYAN "RESULT OF SAMPLING AT THE NEXT BEST POINT (positive improvement is better) WITH OPT HYPERPARMS:\n" OL_ANSI_COLOR_RESET);
    printf("new function value: %.18E, previous best: %.18E, difference (improvement): %.18E\n", function_value, best_so_far, best_so_far - function_value);
  }

  {  // optimize EI using a model with randomly chosen (incorrect) hyperparameters
    // see how we would've done with the wrong hyperparameters
    printf(OL_ANSI_COLOR_CYAN "OPTIMIZING EXPECTED IMPROVEMENT... (wrong hyperparameters) \n" OL_ANSI_COLOR_RESET);

    // choose some wrong hyperparameters
    std::vector<double> hyperparameters_wrong(covariance_original.GetNumberOfHyperparameters());
    boost::uniform_real<double> uniform_double_for_wrong_hyperparameter(0.1, 0.5);
    CovarianceClass covariance_wrong(dim, 1.0, 1.0);
    FillRandomCovarianceHyperparameters(uniform_double_for_wrong_hyperparameter, &uniform_generator,
                                        &hyperparameters_wrong, &covariance_wrong);
    GaussianProcess gp_wrong_hyper(covariance_wrong, points_sampled.data(), points_sampled_value.data(),
                                   noise_variance.data(), dim, num_sampled);

    bool found_flag = false;
    ComputeOptimalPointsToSampleWithRandomStarts(gp_wrong_hyper, gd_params, domain, thread_schedule,
                                                 points_being_sampled.data(), num_to_sample,
                                                 num_being_sampled, best_so_far, max_int_steps,
                                                 &found_flag, &uniform_generator, normal_rng_vec.data(),
                                                 next_point_winner.data());
    printf(OL_ANSI_COLOR_CYAN "EI OPTIMIZATION FINISHED (wrong hyperparameters). Success status: %s\n" OL_ANSI_COLOR_RESET, found_flag ? "True" : "False");
    printf("Next best sample point according to EI (wrong hyper):\n");
    PrintMatrix(next_point_winner.data(), 1, dim);

    // check what the actual improvement would've been by sampling from our GP with wrong hyperparameters and comparing to best_so_far
    // Not sure if this comparison is valid; maybe gp_generator needs to have the same random state for both calls to
    // SamplePointFromGP?  Or maybe it only makes sense to look after repeated calls?  I think putting the PRNG in the same
    // state for both draws is a reasonable comparison, since then we have two identical GPs

    // put gp_generator in same prng state as when we did the draw for the optimized hyperparameter result
    gp_generator.ResetToMostRecentSeed();
    double function_value = gp_generator.SamplePointFromGP(next_point_winner.data(), 0.0);  // sample w/o noise
    printf(OL_ANSI_COLOR_CYAN "RESULT OF SAMPLING AT THE NEXT BEST POINT (positive improvement is better) WITH WRONG HYPERPARAMS:\n" OL_ANSI_COLOR_RESET);
    printf("new function value: %.18E, previous best: %.18E, difference (improvement): %.18E\n", function_value, best_so_far, best_so_far - function_value);
  }

  return 0;
}  // end main
