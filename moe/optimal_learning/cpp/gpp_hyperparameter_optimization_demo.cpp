/*!
  \file gpp_hyperparameter_optimization_demo.cpp
  \rst
  ``moe/optimal_learning/cpp/gpp_hyperparameter_optimization_demo.cpp``

  This is a demo for the model selection (via hyperparameter optimization) capability
  present in this project.  These capabilities live in
  gpp_model_selection.

  In gpp_expected_improvement_demo, we choose the hyperparameters arbitrarily.  Here,
  we will walk through an example of how one would select hyperparameters for a given
  class of covariance function; here, SquareExponential will do.  This demo supports:

  1. User-specified training data
  2. Randomly generated training data (more automatic)

  More details on the second case:

  1. Choose a set of hyperparameters randomly: source covariance
  2. Build a fake\* training set by drawing from a GP with source covariance, at randomly
     chosen locations
     \* By defining OL_USER_INPUTS to 1, you can specify your own input data.
  3. Choose a new random set of hyperparameters and run hyperparameter optimization

     a. Show log likelihood using the optimized hyperparameters AND the source hyperparameters
     b. observe that with larger training sets, the optimized hyperparameters converge
        to the source values; but in smaller sets other optima may exist

  Further notes about [newton] optimization performance and robustness are spread throughout the
  demo code, placed near the function call/object construction that they are relevant to.

  Please read and understand gpp_expected_improvement_demo.cpp before going through
  this example.  In addition, understanding gpp_model_selection.hpp's
  file comments (as well as cpp for devs) is prerequisite.
\endrst*/

#include <cstdio>

#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

#define OL_USER_INPUTS 0

using namespace optimal_learning;  // NOLINT, i'm lazy in this file which has no external linkage anyway

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  // here we set some configurable parameters
  // feel free to change them (and recompile) as you explore
  // comments next to each parameter will indicate its purpose and domain

  // the "spatial" dimension, aka the number of independent (experiment) parameters
  // i.e., this is the dimension of the points in points_sampled
  static const int dim = 3;  // > 0

  // number of points that we have already sampled; i.e., size of the training set
  static const int num_sampled = 100;  // >= 0
  // observe that as num_sampled increases, the optimal set of hyperparameters (via optimization) will approach
  // the set used to generate the input data (in the case of generating inputs randomly from a GP).  Don't try overly
  // large values or it will be slow; for reference 500 samples takes ~2-3 min on my laptop whereas 100 samples takes ~1s

  // the log likelihoods will also decrease in value since by adding more samples, we are more greatly restricting the GP
  // into ever-narrower sets of likely realizations

  UniformRandomGenerator uniform_generator(314);  // repeatable results
  // construct with (base_seed, thread_id) to generate a 'random' seed

  // specifies the domain of each independent variable in (min, max) pairs
  // set appropriately for user-specified inputs
  // mostly irrelevant for randomly generated inputs
  std::vector<ClosedInterval> domain_bounds = {
    {-1.5, 2.3},  // first dimension
    {0.1, 3.1},   // second dimension
    {1.7, 2.9}};  // third dimension
  DomainType domain(domain_bounds.data(), dim);

  // now we allocate point sets; ALL POINTS MUST LIE INSIDE THE DOMAIN!
  std::vector<double> points_sampled(num_sampled*dim);

  std::vector<double> points_sampled_value(num_sampled);

  // default to 0 noise
  std::vector<double> noise_variance(num_sampled, 0.0);  // each entry must be >= 0.0
  // choosing too much noise makes little sense: cannot make useful predicitions if data
  // is drowned out by noise
  // choosing 0 noise is dangerous for large problems; the covariance matrix becomes very
  // ill-conditioned, and adding noise caps the maximum condition number at roughly
  // 1.0/min(noise_variance)

  // covariance selection
  using CovarianceClass = SquareExponential;  // see gpp_covariance.hpp for other options

  // arbitrary hyperparameters used to generate data
  std::vector<double> hyperparameters_original(1 + dim);
  CovarianceClass covariance_original(dim, 1.0, 1.0);
  // CovarianceClass provides SetHyperparameters, GetHyperparameters to read/modify
  // hyperparameters later on
  // Generate hyperparameters randomly
  boost::uniform_real<double> uniform_double_for_hyperparameter(0.5, 1.5);
  FillRandomCovarianceHyperparameters(uniform_double_for_hyperparameter, &uniform_generator,
                                      &hyperparameters_original, &covariance_original);

  std::vector<ClosedInterval> hyperparameter_domain_bounds(covariance_original.GetNumberOfHyperparameters(), {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(),
                                                 covariance_original.GetNumberOfHyperparameters());

  // now fill data
#if OL_USER_INPUTS == 1
  // if you prefer, insert your own data here
  // requirements aka variables that must be set:
  // noise variance, num_sampled values: defaulted to 0; need to set this for larger data sets to deal with conditioning
  // points_sampled, num_sampled*dim values: the locations of already-sampled points; must be INSIDE the domain
  // points_sampled_value, num_sampled values: the function values at the already-sampled points
  // covariance_perturbed: a CovarianceClass object constructed with perturbed (from covariance_original) hyperparameters;
  //   must have decltype(covariance_perturbed) == decltype(covariance_original) for hyperparameter opt to make any sense

  // NOTE: the GP is 0-mean, so shift your points_sampled_value entries accordingly
  // e.g., if the samples are from a function with mean M, subtract it out
#else
  // generate GP inputs randomly

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

  // choose a random initial guess reasonably far away from hyperparameters_original
  // to find some optima (see WARNING2 below), it may be necessary to start with hyperparameters smaller than the originals or
  // of similar magnitude
  std::vector<double> hyperparameters_perturbed(covariance_original.GetNumberOfHyperparameters());
  CovarianceClass covariance_perturbed(dim, 1.0, 1.0);
  boost::uniform_real<double> uniform_double_for_wrong_hyperparameter(5.0, 12.0);
  FillRandomCovarianceHyperparameters(uniform_double_for_wrong_hyperparameter, &uniform_generator,
                                      &hyperparameters_perturbed, &covariance_perturbed);
#endif

  // log likelihood type selection
  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  // log likelihood evaluator object
  LogLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(),
                                           noise_variance.data(), dim, num_sampled);

  int total_newton_errors = 0;  // number of newton runs that failed due to singular hessians
  int newton_max_num_steps = 500;  // max number of newton steps
  double gamma_newton = 1.05;  // newton diagonal dominance scale-down factor (see newton docs for details)
  double pre_mult_newton = 1.0e-1;  // newton diagonal dominance scaling factor (see newton docs for details)
  double max_relative_change_newton = 1.0;
  double tolerance_newton = 1.0e-11;
  NewtonParameters newton_parameters(1, newton_max_num_steps, gamma_newton, pre_mult_newton,
                                     max_relative_change_newton, tolerance_newton);

  // call newton to optimize hyperparameters
  // in general if this takes the full hyperparameter_max_num_steps iterations, something went wrong
  // newton's solution:
  std::vector<double> new_newton_hyperparameters(covariance_original.GetNumberOfHyperparameters());

  printf(OL_ANSI_COLOR_CYAN "ORIGINAL HYPERPARMETERS:\n" OL_ANSI_COLOR_RESET);
  printf("Original Hyperparameters:\n");
  PrintMatrix(hyperparameters_original.data(), 1, covariance_original.GetNumberOfHyperparameters());

  printf(OL_ANSI_COLOR_CYAN "NEWTON OPTIMIZED HYPERPARAMETERS:\n" OL_ANSI_COLOR_RESET);
  // run newton optimization
  total_newton_errors += NewtonHyperparameterOptimization(log_marginal_eval, covariance_perturbed,
                                                          newton_parameters, hyperparameter_domain,
                                                          new_newton_hyperparameters.data());
  // WARNING: the gradient of log marginal appears to go to 0 as you move toward infinity.  if you do not start
  // close enough to an optima or have overly aggressive diagonal dominance settings, newton will skip miss everything
  // going on locally and shoot out to these solutions.
  // Having hyperparameters = 1.0e10 is nonsense, and usually this problem is further signaled by a log marginal likelihood
  // that is POSITIVE (impossible since p \in [0,1], so log(p) \in (-\infty, 0])

  // Long-term, we should solve this problem by multistarting newton.  Additionally there will be some kind of "quick kill"
  // mechanism needed--when newton is wandering down the wrong path (or to an already-known solution?) we should detect it
  // and kill it quickly to keep cost low.
  // For now, just play around with different initial conditions or more conservative gamam settings.

  // WARNING2: for small num_sampled, it often appears that the solution becomes independent of one or more hyperparameters.
  // e.g., in 2D, we'd have an optimal "ridge."  Finding this robustly requires starting near it, so the random choice of
  // initial conditions can fail horribly in general.

  // WARNING3: if you choose large values of num_sampled (like 300), this can be quite slow; about 5min on my computer
  // sometimes the reason is that machine prescision prevents us from reaching the cutoff criterion:
  // norm_gradient_likelihood <= 1.0e-13 in NewtonHyperparameterOptimization() in gpp_model_selection...cpp
  // So you may need to relax this to 1.0e-10 or something so that we aren't just spinning wheels at almost-converged but
  // unable to actually move anywhere.

  printf("Result of newton:\n");
  PrintMatrix(new_newton_hyperparameters.data(), 1, covariance_original.GetNumberOfHyperparameters());

  if (total_newton_errors > 0) {
    printf("WARNING: %d newton runs exited due to singular Hessian matrices.\n", total_newton_errors);
  }

  printf(OL_ANSI_COLOR_CYAN "LOG LIKELIHOOD + GRADIENT AT NEWTON OPTIMIZED FINAL HYPERPARAMS:\n" OL_ANSI_COLOR_RESET);

  CovarianceClass covariance_final(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
  typename LogLikelihoodEvaluator::StateType log_marginal_state_newton_optimized_hyper(log_marginal_eval,
                                                                                       covariance_final);
  double newton_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_newton_optimized_hyper);
  printf("newton optimized log marginal likelihood = %.18E\n", newton_log_marginal_opt);

  std::vector<double> grad_log_marginal_opt(covariance_final.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_newton_optimized_hyper,
                                             grad_log_marginal_opt.data());
  printf("grad log likelihood: ");
  PrintMatrix(grad_log_marginal_opt.data(), 1, covariance_final.GetNumberOfHyperparameters());

  printf(OL_ANSI_COLOR_CYAN "LOG LIKELIHOOD + GRADIENT AT ORIGINAL HYPERPARAMS:\n" OL_ANSI_COLOR_RESET);
  typename LogLikelihoodEvaluator::StateType log_marginal_state_original_hyper(log_marginal_eval,
                                                                               covariance_original);

  double original_log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_original_hyper);
  printf("original log marginal likelihood = %.18E\n", original_log_marginal);

  std::vector<double> original_grad_log_marginal(covariance_original.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_original_hyper,
                                             original_grad_log_marginal.data());
  printf("grad log likelihood: ");
  PrintMatrix(original_grad_log_marginal.data(), 1, covariance_original.GetNumberOfHyperparameters());

  return 0;
}  // end main
