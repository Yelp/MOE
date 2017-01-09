/*!
  \file gpp_model_selection.hpp
  \rst
  Table of Contents:

  1. FILE OVERVIEW
  2. MODEL SELECTION OVERVIEW
  3. LOG LIKELIHOOD METRICS OF MODEL QUALITY

     a. LOG MARGINAL LIKELIHOOD (LML)
     b. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)

  4. HYPERPARAMETER OPTIMIZATION OF LOG LIKELIHOOD
  5. IMPLEMENTATION NOTES

  **1. FILE OVERVIEW**

  As a preface, you should read gpp_math.hpp's comments first (if not also gpp_math.cpp) to get an overview
  of Gaussian Processes (GPs) and how we are using them (Expected Improvement, EI).

  .. NOTE:: These comments have been copied into interfaces/log_likelihood_interface.py (file comments) and
    cpp_wrappers/log_likelihood.py (LogMarginalLikelihood and LeaveOneOutLogLikelihood class comments).

  This file deals with model selection via hyperparameter optimization, as the name implies.  In our discussion of GPs,
  we did not pay much attention to the underlying covariance function.  We noted that the covariance is extremely
  important since it encodes our assumptions about the objective function f(x) that we are trying to learn; i.e.,
  the covariance describes the nearness/similarity of points in the input space.  Also, the GP was clearly indicated
  to be a function of the covariance, but we simply assumed that the selection of covariance was an already-solved
  problem (or even worse, arbitrary!).

  To tackle the model selection problem, this file provides some classes that encapsulate log likelihood measures
  of model quality:

  * LogMarginalLikelihoodEvaluator
  * LeaveOneOutLogLikelihoodEvaluator

  both of which follow the Evaluator/State "idiom" described in gpp_common.hpp.

  For selecting the best hyperparameters, this file provides two multistart optimization wrappers
  for gradient descent and Newton, that maximize the previous log likelihood measures:

  * MultistartGradientDescentHyperparameterOptimization<LogLikelihoodEvaluator, Domain>()
  * MultistartNewtonHyperparameterOptimization<LogLikelihoodEvaluator, Domain>()

  These functions are wrappers for templated code in gpp_optimization.hpp.  The wrappers just set up inputs for use
  with the routines in gpp_optimization.hpp.  These are the preferred endpoints for hyperparameter optimization.

  If these derivative-based techniques fail, then we also have simple 'dumb' search fallbacks:

  * LatinHypercubeSearchHyperparameterOptimization<LogLikelihoodEvaluator>()  (eval log-likelihood at random points)
  * EvaluateLogLikelihoodAtPointList<LogLikelihoodEvaluator, Domain>()  (eval log-likelihood at specified points)

  This file also provides single-start versions of each optimization technique:

  * RestartedGradientDescentHyperparameterOptimization<LogLikelihoodEvaluator, Domain>()
  * NewtonHyperparameterOptimization<LogLikelihoodEvaluator, Domain>()

  Typically these will not be called directly.

  To better understand model selection, let's look at a common covariance used in our computation, square exponential:

  ``cov(x_1, x_2) = \alpha * \exp(-0.5*r^2), where r = \sum_{i=1}^d (x_1_i - x_2_i)^2 / L_i^2``.

  Here, ``\alpha`` is ``\sigma_f^2``, the signal variance, and the ``L_i`` are length scales.  The vector ``[\alpha, L_1, ... , L_d]``
  are called the "hyperparameters" or "free parameters" (see gpp_covariance.hpp for more details).  There is nothing in
  the covariance  that guides the choice of the hyperparameters; ``L_1 = 0.001`` is just as valid as ``L_1 = 1000.0.``

  Clearly, the value of the covariance changes substantially if ``L_i`` varies by a factor of two, much less 6 orders of
  magnitude.  That is the difference between saying variations of size \approx 1.0 in x_i, the first spatial dimension,
  are extremely important vs almost irrelevant.

  So how do we know what hyperparameters to choose?  This question/problem is more generally called "Model Selection."
  Although the problem is far from solved, we will present the approaches implemented here; as usual, we use
  Rasmussen & Williams (Chapter 5 now) as a guide/reference.

  However, we will not spend much time discussing selection across different classes of covariance functions; e.g.,
  Square Exponential vs Matern w/various ``\nu``, etc.  We have yet to develop any experience/intuition with this problem
  and are temporarily punting it.  For now, we follow the observation in Rasmussen & Williams that Square Exponential
  is a popular choice and appears to work very well.  (This is still a very important problem; e.g., there may be
  scenarios when we would prefer a non-stationary or periodic covariance, and the methods discussed here do not cover
  this aspect of selection.  Such covariance options are not yet implemented though.)

  We do note that the techniques for selecting covariance classes more or less require hyperparameter optimization
  on each individual covariance.  The likely approach would be to produce the best fit (according to chosen metrics)
  using each type of covariance (using optimization) and then choose the best performer across the group.

  Following R&W, the following discussion:

  1. Provides some more concerete description/overview of the model selection problem
  2. Discusses the two metrics for model evaluation that we currently have
  3. Discusses the currently implemented optimization techniques for performing hyperparameter optimization

  **2. MODEL SELECTION OVERVIEW**

  Generally speaking, there are a great many tunable parameters in any model-based learning algorithm.  In our case,
  the GP takes a covariance function as input; the selection of the covariance class as well as the choice of hyperparameters
  are all part of the model selection process.  Determining these details of the [GP] model is the model selection problem.

  In order to evaluate the quality of models (and solve model selction), we need some kind of metric.  The literature suggests
  too many to cite, but R&W groups them into three common approaches (5.1, p108):

  A. compute the probability of the model given the data (e.g., LML)
  B. estimate the genereralization error (e.g., LOO-CV)
  C. bound the generalization error

  where "generalization error" is defined as "the average error on unseen test examples (from the same distribution
  as the training cases)."  So it's a measure of how well or poorly the model predicts reality.  This idea will be more
  clear in the LOO-CV discussion.  Let's dive into that next.

  **3. LOG LIKELIHOOD METRICS OF MODEL QUALITY**

  **3a. LOG MARGINAL LIKELIHOOD (LML)**

  (Rasmussen & Williams, 5.4.1)
  The Log Marginal Likelihood measure comes from the ideas of Bayesian model selection, which use Bayesian inference
  to predict distributions over models and their parameters.  The cpp file comments explore this idea in more depth.
  For now, we will simply state the relevant result.  We can build up the notion of the "marginal likelihood":
  probability(observed data GIVEN sampling points (``X``), model hyperparameters, model class (regression, GP, etc.)),
  which is denoted: ``p(y | X, \theta, H_i)`` (see the cpp file comments for more).

  So the marginal likelihood deals with computing the probability that the observed data was generated from (another
  way: is easily explainable by) the given model.

  The marginal likelihood is in part paramaterized by the model's hyperparameters; e.g., as mentioned above.  Thus
  we can search for the set of hyperparameters that produces the best marginal likelihood and use them in our model.
  Additionally, a nice property of the marginal likelihood optimization is that it automatically trades off between
  model complexity and data fit, producing a model that is reasonably simple while still explaining the data reasonably
  well.  See the cpp file comments for more discussion of how/why this works.

  In general, we do not want a model with perfect fit and high complexity, since this implies overfit to input noise.
  We also do not want a model with very low complexity and poor data fit: here we are washing the signal out with
  (assumed) noise, so the model is simple but it provides no insight on the data.

  This is not magic.  Using GPs as an example, if the covariance function is completely mis-specified, we can blindly
  go through with marginal likelihood optimization, obtain an "optimal" set of hyperparameters, and proceed... never
  realizing that our fundamental assumptions are wrong.  So care is always needed.

  **3b. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)**

  (Rasmussen & Williams, Chp 5.4.2)
  In cross validation, we split the training data, ``X``, into two sets--a sub-training set and a validation set.  Then we
  train a model on the sub-training set and test it on the validation set.  Since the validation set comes from the
  original training data, we can compute the error.  In effect we are examining how well the model explains itself.

  Leave One Out CV works by considering n different validation sets, one at a time.  Each point of ``X`` takes a turn
  being the sole member of the validation set.  Then for each validation set, we compute a log pseudo-likelihood, measuring
  how probable that validation set is given the remaining training data and model hyperparameters.

  Again, we can maximize this quanitity over hyperparameters to help us choose the "right" set for the GP.

  **4. HYPERPARAMETER OPTIMIZATION OF LOG LIKELIHOOD**

  Now that we have discussed the Log Marginal Likelihood and Leave One Out Cross Validation log pseudo-likelihood measures
  of model quality, what do we do with them?  How do they help us choose hyperparameters?

  From here, we can apply anyone's favorite optimization technique to maximize log likelihoods wrt hyperparameters.  The
  hyperparameters that maximize log likelihood provide the model configuration that is most likely to have produced the
  data observed so far, ``(X, f)``.

  In principle, this approach always works.  But in practice it is often not that simple.  For example, suppose the underlying
  objective is periodic and we try to optimize hyperparameters for a class of covariance functions that cannot account
  for the periodicity.  We can always\* find the set of hyperparameters that maximize our chosen log likelihood measure
  (LML or LOO-CV), but if the covariance is mis-specified or we otherwise make invalid assumptions about the objective
  function, then the results are not meaningful at best and misleading at worst.  It becomes a case of garbage in,
  garbage out.

  \* Even this is tricky.  Log likelihood is almost never a convex function.  For example, with LML + GPs, you often expect
  at least two optima, one more complex solution (short length scales, less intrinsic noise) and one less complex
  solution (longer length scales, higher intrinsic noise).  There are even cases where no optima (to machine precision)
  exist or cases where solutions lie on (lower-dimensional) manifold(s) (e.g., locally the likelihood is (nearly) independent
  of one or more hyperparameters).

  **5. IMPLEMENTATION NOTES**

  a. This file has a few primary endpoints for model selection (aka hyperparameter optimization):

     i. LatinHypercubeSearchHyperparameterOptimization<>():

        Takes in a ``log_likelihood_evaluator`` describing the prior, covariance, domain, config, etc.;
        searches over a set of (random) hyperparameters and outputs the set producing the best model fit.

     ii. MultistartGradientDescentHyperparameterOptimization<>():

         Takes in a ``log_likelihood_evaluator`` describing the prior, covariance, domain, config, etc.;
         searches for the best hyperparameters (of covariance) using multiple gradient descent runs.

         Single start version available in: RestartedGradientDescentHyperparameterOptimization<>().

     iii. MultistartNewtonHyperparameterOptimization<>() (Recommended):

          Takes in a ``log_likelihood_evaluator`` describing the prior, covariance, domain, config, etc.;
          searches for the best hyperparameters (of covariance) using multiple Newton runs.

          Single start version available in: NewtonHyperparameterOptimization<>().

     .. NOTE::
         See ``gpp_model_selection.cpp``'s header comments for more detailed implementation notes.

         There are also several other functions with external linkage in this header; these
         are provided primarily to ease testing and to permit lower level access from python.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_HPP_

#include <cmath>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

/*!\rst
  Enum for the various log likelihood measures. Convenient for specifying which log
  likelihood to use in testing and also used by the Python interface to specify which
  log likelihood measure to optimize in hyperparameter optimization.
\endrst*/
enum class LogLikelihoodTypes {
  //! LogMarginalLikelihoodEvaluator
  kLogMarginalLikelihood = 0,
  //! LeaveOneOutLogLikelihoodEvaluator
  kLeaveOneOutLogLikelihood = 1,
};

struct UniformRandomGenerator;
struct LogMarginalLikelihoodState;
struct LeaveOneOutLogLikelihoodState;

/*!\rst
  This serves as a quick summary of the Log Marginal Likelihood (LML).  Please see the file comments here and
  in the corresponding .cpp file for further details.

  Class for computing the Log Marginal Likelihood.  Given a particular covariance function (including hyperparameters) and
  training data ((point, function value, measurement noise) tuples), the log marginal likelihood is the log probability that
  the data were observed from a Gaussian Process would have generated the observed function values at the given measurement
  points.  So log marginal likelihood tells us "the probability of the observations given the assumptions of the model."
  Log marginal sits well with the Bayesian Inference camp.
  (Rasmussen & Williams p118)

  This quantity primarily deals with the trade-off between model fit and model complexity.  Handling this trade-off is automatic
  in the log marginal likelihood calculation.  See Rasmussen & Williams 5.2 and 5.4.1 for more details.

  We can use the log marginal likelihood to determine how good our model is.  Additionally, we can maximize it by varying
  hyperparameters (or even changing covariance functions) to improve our model quality.  Hence this class provides access
  to functions for computing log marginal likelihood and its hyperparameter gradients.

  .. Note:: These class comments are duplicated in Python: cpp_wrappers.log_likelihood.LogMarginalLikelihood
\endrst*/
class LogMarginalLikelihoodEvaluator final {
 public:
  //! string name of this log likelihood evaluator for logging
  constexpr static char const * kName = "log_marginal_likelihood";

  using StateType = LogMarginalLikelihoodState;

  /*!\rst
    Constructs a LogMarginalLikelihoodEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
      :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
      :points_sampled[dim][num_sampled]: points that have already been sampled
      :points_sampled_value[num_sampled]: values of the already-sampled points
      :noise_variance[num_sampled]: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_sampled: number of already-sampled points
  \endrst*/
  LogMarginalLikelihoodEvaluator(double const * restrict points_sampled_in,
                                 double const * restrict points_sampled_value_in,
                                 double const * restrict noise_variance_in,
                                 int dim_in, int num_sampled_in) OL_NONNULL_POINTERS;

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_sampled() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_sampled_;
  }

  /*!\rst
    Wrapper for ComputeLogLikelihood(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * log_likelihood_state) const noexcept OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeLogLikelihood(*log_likelihood_state);
  }

  /*!\rst
    Wrapper for ComputeGradLogLikelihood(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * log_likelihood_state,
                                    double * restrict grad_log_marginal) const noexcept OL_NONNULL_POINTERS {
    ComputeGradLogLikelihood(log_likelihood_state, grad_log_marginal);
  }

  /*!\rst
    Wrapper for ComputeHessianLogLikelihood(); see that function for details.
  \endrst*/
  void ComputeHessianObjectiveFunction(StateType * log_likelihood_state,
                                       double * restrict hessian_log_marginal) const noexcept OL_NONNULL_POINTERS {
    ComputeHessianLogLikelihood(log_likelihood_state, hessian_log_marginal);
  }

  /*!\rst
    Sets up the LogMarginalLikelihoodState object so that it can be used to compute log marginal and its gradients.
    ASSUMES all needed space is ALREADY ALLOCATED.

    This function should not be called directly; instead use LogMarginalLikelihoodState::SetupState.

    \param
      :log_likelihood_state[1]: constructed state object with appropriate sized allocations
    \output
      :log_likelihood_state[1]: fully configured state object, ready for use by this class's member functions
  \endrst*/
  void FillLogLikelihoodState(StateType * log_likelihood_state) const OL_NONNULL_POINTERS;

  /*!\rst
    Computes the log marginal likelihood, ``log(p(y | X, \theta))``.
    That is, the probability of observing the training values, ``y``, given the training points, ``X``,
    and hyperparameters (of the covariance function), ``\theta``.

    This is a measure of how likely it is that the observed values came from our Gaussian Process Prior.

    \param
      :log_likelihood_state: properly configured state oboject
    \return
      natural log of the marginal likelihood of the GP model
  \endrst*/
  double ComputeLogLikelihood(const StateType& log_likelihood_state) const noexcept OL_WARN_UNUSED_RESULT;

  /*!\rst
    Computes the (partial) derivatives of the log marginal likelihood with respect to each hyperparameter of our covariance function.

    Let ``n_hyper = covariance_ptr->GetNumberOfHyperparameters();``

    \param
      :log_likelihood_state[1]: properly configured state oboject
    \output
      :log_likelihood_state[1]: state with temporary storage modified
      :grad_log_marginal[n_hyper]: gradient of log marginal likelihood wrt each hyperparameter of covariance
  \endrst*/
  void ComputeGradLogLikelihood(StateType * log_likelihood_state,
                                double * restrict grad_log_marginal) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Constructs the Hessian matrix of the log marginal likelihood function.  This matrix is symmetric.  It is also
    negative definite near maxima of the log marginal.

    See HyperparameterHessianCovariance() docs in CovarianceInterface (gpp_covariance.hpp) for details on the structure
    of the Hessian matrix.

    \param
      :log_likelihood_state[1]: properly configured state oboject
    \output
      :log_likelihood_state[1]: state with temporary storage modified
      :hessian_log_marginal[n_hyper][n_hyper]: ``(i,j)``-th entry is ``\mixpderiv{LML}{\theta_i}{\theta_j}``, where ``LML = log(p(y | X, \theta))``
  \endrst*/
  void ComputeHessianLogLikelihood(StateType * log_likelihood_state,
                                   double * restrict hessian_log_marginal) const noexcept OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(LogMarginalLikelihoodEvaluator);

 private:
  /*!\rst
    Constructs the tensor of gradients (wrt hyperparameters) of the covariance function at all pairs of ``points_sampled_``.

    The result is stored in ``state->grad_hyperparameter_cov_matrix``.  So we are computing ``\pderiv{cov(X_i, X_j)}{\theta_k}``.  These
    data are ordered as: ``grad_hyperparameter_cov_matrix[i][j][k]`` (i.e., ``num_hyperparmeters`` matrices of size ``Square(num_sampled_)``).

    .. Note:: ``grad_hyperparameter_cov_matrix[i][j][k] == grad_hyperparameter_cov_matrix[j][i][k]``

    \param
      :log_likelihood_state[1]: properly configured state object
    \output
      :log_likelihood_state[1]: state with grad_hyperparameter_cov_matrix filled
  \endrst*/
  void BuildHyperparameterGradCovarianceMatrix(StateType * log_likelihood_state) const noexcept;

  /*!\rst
    Constructs the tensor of hessians (wrt hyperparameters) of the covariance function at all pairs of ``points_sampled_``.
    The result is ``\mixpderiv{cov(X_i, X_j)}{\theta_k}{\theta_l}``, stored in ``hessian_hyperparameter_cov_matrix[i][j][k][l]``.

    .. Note:: this tensor has several symmetries: ``A[i][j][k][l] == A[j][i][k][l]`` and ``A[i][j][k][l] == A[i][j][l][k]``.

    \param
      :log_likelihood_state[1]: properly configured state object
    \output
      :log_likelihood_state[1]: state object with temporary storage modified
      :hessian_hyperparameter_cov_matrix[num_sampled][num_sampled][n_hyper][n_hyper]:
        ``(i,j,k,l)``-th entry is ``\mixpderiv{cov(X_i, X_j)}{\theta_k}{\theta_l}``
  \endrst*/
  void BuildHyperparameterHessianCovarianceMatrix(StateType * log_likelihood_state,
                                                  double * hessian_hyperparameter_cov_matrix) const noexcept;

  // size information
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! number of points in points_sampled
  int num_sampled_;

  // state variables
  //! coordinates of already-sampled points, X
  std::vector<double> points_sampled_;
  //! function values at points_sampled, y
  std::vector<double> points_sampled_value_;
  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
};

/*!\rst
  State object for LogMarginalLikelihoodEvaluator.  This object tracks the covariance object as well as derived quantities
  that (along with the training points/values in the Evaluator class) fully specify the log marginal likelihood.  Since this
  is used to optimize the log marginal likelihood, the covariance's hyperparameters are variable.

  See general comments on State structs in gpp_common.hpp's header docs.
\endrst*/
struct LogMarginalLikelihoodState final {
  using EvaluatorType = LogMarginalLikelihoodEvaluator;

  /*!\rst
    Constructs a LogMarginalLikelihoodState object with a specified covariance object (in particular, new hyperparameters).
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that the Evaluator can be used to compute log marginal likelihood, gradients, etc.

    .. WARNING:: This object's state is INVALIDATED if the log_likelihood_eval used in construction is mutated!
      SetupState() should be called again in such a situation.

    \param
      :log_likelihood_eval: LogMarginalLikelihoodEvaluator object that this state is being used with
      :covariance_in: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
  \endrst*/
  LogMarginalLikelihoodState(const EvaluatorType& log_likelihood_eval, const CovarianceInterface& covariance_in);

  LogMarginalLikelihoodState(LogMarginalLikelihoodState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_hyperparameters;
  }

  void SetCurrentPoint(const EvaluatorType& log_likelihood_eval,
                          double const * restrict hyperparameters) OL_NONNULL_POINTERS {
    SetHyperparameters(log_likelihood_eval, hyperparameters);
  }

  void GetCurrentPoint(double * restrict hyperparameters) OL_NONNULL_POINTERS {
    GetHyperparameters(hyperparameters);
  }

  /*!\rst
    Get hyperparameters of underlying covariance function.

    \output
      :hyperparameters[num_hyperparameters]: covariance's hyperparameters
  \endrst*/
  void GetHyperparameters(double * restrict hyperparameters) const noexcept OL_NONNULL_POINTERS {
    covariance_ptr->GetHyperparameters(hyperparameters);
  }

  /*!\rst
    Change the hyperparameters of the underlying covariance function.
    Update the state's derived quantities to be consistent with the new hyperparameters.

    \param
      :log_likelihood_eval: LogMarginalLikelihoodEvaluator object that this state is being used with
      :hyperparameters[num_hyperparameters]: hyperparameters to change to
  \endrst*/
  void SetHyperparameters(const EvaluatorType& log_likelihood_eval,
                             double const * restrict hyperparameters) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with new hyperparameters.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables for log likelihood (+ gradient) evaluation.

    .. WARNING:: This object's state is INVALIDATED if the log_likelihood used in SetupState is mutated!
      SetupState() should be called again in such a situation.

    \param
      :log_likelihood_eval: log likelihood evaluator object that describes the training/already-measured data
      :hyperparameters[num_hyperparameters]: hyperparameters to change to
  \endrst*/
  void SetupState(const EvaluatorType& log_likelihood_eval,
                  double const * restrict hyperparameters) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim;
  //! number of points in points_sampled
  int num_sampled;
  //! number of hyperparameters of covariance; i.e., covariance_ptr->GetNumberOfHyperparameters()
  int num_hyperparameters;

  // state variables
  //! covariance class (for computing covariance and its gradients)
  std::unique_ptr<CovarianceInterface> covariance_ptr;

  // derived variables
  //! cholesky factorization of ``K``
  std::vector<double> K_chol;
  //! ``K^-1 * y``; computed WITHOUT forming ``K^-1``
  std::vector<double> K_inv_y;

  // temporary storage: preallocated space used by LogMarginalLikelihoodEvaluator's member functions
  //! ``\pderiv{K_{ij}}{\theta_k}``; temporary b/c it is overwritten with each computation of GradLikelihood
  std::vector<double> grad_hyperparameter_cov_matrix;
  //! temporary storage space of size ``num_sampled``
  std::vector<double> temp_vec;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(LogMarginalLikelihoodState);
};

/*!\rst
  This serves as a quick summary of Leave One Out Cross Validation (LOO-CV).  Please see the file comments here and
  in the corresponding .cpp file for further details.

  Class for computing the Leave-One-Out Cross Validation (LOO-CV) Log Pseudo-Likelihood.  Given a particular covariance
  function (including hyperparameters) and training data ((point, function value, measurement noise) tuples), the log
  LOO-CV pseudo-likelihood expresses how well the model explains itself.

  That is, cross validation involves splitting the training set into a sub-training set and a validation set.  Then we measure
  the log likelihood that a model built on the sub-training set could produce the values in the validation set.

  Leave-One-Out CV does this process ``|y|`` times: on the ``i``-th run, the sub-training set is ``(X, y)`` with the ``i``-th point removed
  and the validation set is the ``i``-th point.  Then the predictive performance of each sub-model are aggregated into a
  psuedo-likelihood.

  This quantity primarily deals with the internal consistency of the model--how well it explains itself.  The LOO-CV
  likelihood gives an "estimate for the predictive probability, whether or not the assumptions of the model may be
  fulfilled." It is a more frequentist view of model selection. (Rasmussen & Williams p118)
  See Rasmussen & Williams 5.3 and 5.4.2 for more details.

  As with the log marginal likelihood, we can use this quantity to measure the performance of our model.  We can also
  maximize it (via hyperparameter modifications or covariance function changes) to improve model performance.
  It has also been argued that LOO-CV is better at detecting model mis-specification (e.g., wrong covariance function)
  than log marginal measures (Rasmussen & Williams p118).

  .. Note:: These class comments are duplicated in Python: cpp_wrappers.log_likelihood.LeaveOneOutLogLikelihood
\endrst*/
class LeaveOneOutLogLikelihoodEvaluator final {
 public:
  //! string name of this log likelihood evaluator for logging
  constexpr static char const * kName = "leave_one_out_log_likelihood";

  using StateType = LeaveOneOutLogLikelihoodState;

  /*!\rst
    Constructs a LeaveOneOutLogLikelihoodEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
      :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
      :points_sampled[dim][num_sampled]: points that have already been sampled
      :points_sampled_value[num_sampled]: values of the already-sampled points
      :noise_variance[num_sampled]: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_sampled: number of already-sampled points
  \endrst*/
  LeaveOneOutLogLikelihoodEvaluator(double const * restrict points_sampled_in,
                                    double const * restrict points_sampled_value_in,
                                    double const * restrict noise_variance_in,
                                    int dim_in, int num_sampled_in) OL_NONNULL_POINTERS;

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_sampled() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_sampled_;
  }

  /*!\rst
    Wrapper for ComputeLogLikelihood(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * log_likelihood_state) const noexcept OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeLogLikelihood(*log_likelihood_state);
  }

  /*!\rst
    Wrapper for ComputeGradLogLikelihood(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * log_likelihood_state,
                                    double * restrict grad_loo) const noexcept OL_NONNULL_POINTERS {
    ComputeGradLogLikelihood(log_likelihood_state, grad_loo);
  }

  /*!\rst
    Wrapper for ComputeHessianLogLikelihood(); see that function for details.
  \endrst*/
  void ComputeHessianObjectiveFunction(StateType * log_likelihood_state,
                                       double * restrict hessian_loo) const OL_NONNULL_POINTERS {
    ComputeHessianLogLikelihood(log_likelihood_state, hessian_loo);
  }

  /*!\rst
    Sets up the LeaveOneOutLogLikelihoodState object so that it can be used to compute log marginal and its gradients.
    ASSUMES all needed space is ALREADY ALLOCATED.

    This function should not be called directly; instead use LeaveOneOutLogLikelihoodState::SetupState.

    \param
      :log_likelihood_state[1]: constructed state object with appropriate sized allocations
    \output
      :log_likelihood_state[1]: fully configured state object, ready for use by this class's member functions
  \endrst*/
  void FillLogLikelihoodState(StateType * log_likelihood_state) const OL_NONNULL_POINTERS;

  /*!\rst
    Computes the log LOO-CV pseudo-likelihood
    That is, split the training data ``(X, y)`` into ``|y|`` training set groups, where in the i-th group, the validation set is
    the ``i``-th point of ``(X, y)`` and the training set is ``(X, y)`` with the ``i``-th point removed.
    Then this likelihood measures the aggregate performance of the ability of a model built on each "leave one out"
    training set to predict the corresponding validation set.  So in some sense it is a measure of model consitency, ensuring
    that we do not perform well on a few points while doing horribly on the others.

    \param
      :log_likelihood_state: properly configured state oboject
    \return
      natural log of the leave one out cross validation pseudo-likelihood of the GP model
  \endrst*/
  double ComputeLogLikelihood(const StateType& log_likelihood_state) const noexcept OL_WARN_UNUSED_RESULT;

  /*!\rst
    Computes the (partial) derivatives of the leave-one-out cross validation log pseudo-likelihood with respect to each hyperparameter of our covariance function.

    Let ``n_hyper = covariance_ptr->GetNumberOfHyperparameters();``

    \param
      :log_likelihood_state[1]: properly configured state oboject
    \output
      :log_likelihood_state[1]: state with temporary storage modified
      :grad_loo[n_hyper]: gradient of leave one out cross validation log likelihood wrt each hyperparameter of covariance
  \endrst*/
  void ComputeGradLogLikelihood(StateType * log_likelihood_state,
                                double * restrict grad_loo) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    NOT IMPLEMENTED.
    Kludge to make it so that I can instantiate MultistartNewtonOptimization<> with LeaveOneOutLogLikelihoodEvaluator in
    gpp_python.cpp. It is an error to select NewtonOptimization with LeaveOneOutLogLikelihoodEvaluator, but I can't find a nicer
    way to generate this error while still being able to treat MultistartNewtonOptimization<> generically.
  \endrst*/
  void ComputeHessianLogLikelihood(StateType * log_likelihood_state,
                                   double * restrict hessian_loo) const OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(LeaveOneOutLogLikelihoodEvaluator);

 private:
  /*!\rst
    Constructs the tensor of gradients (wrt hyperparameters) of the covariance function at all pairs of ``points_sampled_``.

    The result is stored in ``state->grad_hyperparameter_cov_matrix``.  So we are computing ``\pderiv{cov(X_i, X_j)}{\theta_k``}.  These
    data are ordered as: ``grad_hyperparameter_cov_matrix[i][j][k]`` (i.e., ``num_hyperparmeters`` matrices of size ``Square(num_sampled_)``).

    .. Note:: ``grad_hyperparameter_cov_matrix[i][j][k] == grad_hyperparameter_cov_matrix[j][i][k]``

    \param
      :log_likelihood_state[1]: properly configured state object
    \output
      :log_likelihood_state[1]: state with grad_hyperparameter_cov_matrix filled
  \endrst*/
  void BuildHyperparameterGradCovarianceMatrix(StateType * log_likelihood_state) const noexcept;

  // size information
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! number of points in points_sampled
  int num_sampled_;

  // state variables
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
};

/*!\rst
  State object for LeaveOneOutLogLikelihoodEvaluator.  This object tracks the covariance object as well as derived quantities
  that (along with the training points/values in the Evaluator class) fully specify the log marginal likelihood.  Since this
  is used to optimize the log marginal likelihood, the covariance's hyperparameters are variable.

  See general comments on State structs in gpp_common.hpp's header docs.
\endrst*/
struct LeaveOneOutLogLikelihoodState final {
  using EvaluatorType = LeaveOneOutLogLikelihoodEvaluator;

  /*!\rst
    Constructs a LeaveOneOutLogLikelihoodState object with a specified covariance object (in particular, new hyperparameters).
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that the Evaluator can be used to compute log marginal likelihood, gradients, etc.

    .. WARNING:: This object's state is INVALIDATED if the log_likelihood_eval used in construction is mutated!
      SetupState() should be called again in such a situation.

    \param
      :log_likelihood_eval: LogMarginalLikelihoodEvaluator object that this state is being used with
      :covariance_in: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
  \endrst*/
  LeaveOneOutLogLikelihoodState(const EvaluatorType& log_likelihood_eval, const CovarianceInterface& covariance_in);

  LeaveOneOutLogLikelihoodState(LeaveOneOutLogLikelihoodState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_hyperparameters;
  }

  void SetCurrentPoint(const EvaluatorType& log_likelihood_eval,
                          double const * restrict hyperparameters) OL_NONNULL_POINTERS {
    SetHyperparameters(log_likelihood_eval, hyperparameters);
  }

  void GetCurrentPoint(double * restrict hyperparameters) OL_NONNULL_POINTERS {
    GetHyperparameters(hyperparameters);
  }

  /*!\rst
    Get hyperparameters of underlying covariance function.

    \output
      :hyperparameters[num_hyperparameters]: covariance's hyperparameters
  \endrst*/
  void GetHyperparameters(double * restrict hyperparameters) const noexcept OL_NONNULL_POINTERS {
    covariance_ptr->GetHyperparameters(hyperparameters);
  }

  /*!\rst
    Change the hyperparameters of the underlying covariance function.
    Update the state's derived quantities to be consistent with the new hyperparameters.

    \param
      :log_likelihood_eval: LeaveOneOutLogLikelihoodEvaluator object that this state is being used with
      :hyperparameters[num_hyperparameters]: hyperparameters to change to
  \endrst*/
  void SetHyperparameters(const EvaluatorType& log_likelihood_eval,
                             double const * restrict hyperparameters) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with new hyperparameters.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables for log likelihood (+ gradient) evaluation.

    .. WARNING:: This object's state is INVALIDATED if the log_likelihood used in SetupState is mutated!
      SetupState() should be called again in such a situation.

    \param
      :log_likelihood_eval: log likelihood evaluator object that describes the training/already-measured data
      :hyperparameters[num_hyperparameters]: hyperparameters to change to
  \endrst*/
  void SetupState(const EvaluatorType& log_likelihood_eval,
                  double const * restrict hyperparameters) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim;
  //! number of points in points_sampled
  int num_sampled;
  //! number of hyperparameters of covariance; i.e., covariance_ptr->GetNumberOfHyperparameters()
  int num_hyperparameters;

  // state variables
  //! covariance class (for computing covariance and its gradients)
  std::unique_ptr<CovarianceInterface> covariance_ptr;

  // derived variables
  //! cholesky factorization of ``K``
  std::vector<double> K_chol;
  //! ``K^-1``
  std::vector<double> K_inv;
  //! ``K^-1 * y``; computed WITHOUT forming ``K^-1``
  std::vector<double> K_inv_y;

  // temporary storage: preallocated space used by LeaveOneOutLogLikelihoodEvaluator's member functions
  //! ``\pderiv{K_{ij}}{\theta_k}``; temporary b/c it is overwritten with each computation of GradLikelihood
  std::vector<double> grad_hyperparameter_cov_matrix;
  //! temporary: ``K^-1 * grad_hyperparameter_cov_matrix * K^-1 * y``
  std::vector<double> Z_alpha;
  //! temporary: ``K^-1 * grad_hyperparameter_cov_matrix * K^-1``
  std::vector<double> Z_K_inv;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(LeaveOneOutLogLikelihoodState);
};

/*!\rst
  Converts ``domain_bounds`` input from ``log10``-space to linear-space.
  Uniformly samples ``num_multistarts`` initial guesses from the ``log10``-space domain and converts them all to linear space.

  This is a utility function just for reducing code duplication.

  .. Note:: the domain here must be specified in LOG-10 SPACE!

  \param
    :num_hyperparameters: dimension of the domain
    :num_multistarts: number of random points to draw
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :domain_bounds[1]: ``std::vector<ClosedInterval>`` with ``>= num_hyperparameters`` elements specifying
      the boundaries of a n_hyper-dimensional tensor-product domain
      Specify in LOG-10 SPACE!
    :initial_guesses[1]: ``std::vector<double>`` with ``>= num_hyperparameters*num_multistarts`` elements.
      will be overwritten; ordered data[num_hyperparameters][num_multistarts]
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :domain_bounds[1]: overwritten with the domain bounds in linear space
    :initial_guesses[1]: overwritten with num_multistarts points sampled uniformly from the log10-space domain
\endrst*/
inline OL_NONNULL_POINTERS void ConvertFromLogToLinearDomainAndBuildInitialGuesses(
    int num_hyperparameters,
    int num_multistarts,
    UniformRandomGenerator * uniform_generator,
    std::vector<ClosedInterval> * restrict domain_bounds,
    std::vector<double> * restrict initial_guesses) {
  ComputeLatinHypercubePointsInDomain(domain_bounds->data(), num_hyperparameters, num_multistarts,
                                      uniform_generator, initial_guesses->data());

  // exponentiate since domain_bounds is specified in log space
  for (auto& point : *initial_guesses) {
    point = std::pow(10.0, point);
  }
  // domain in linear-space
  for (auto& interval : *domain_bounds) {
    interval = {std::pow(10.0, interval.min), std::pow(10.0, interval.max)};
  }
}

/*!\rst
  Set up state vector.

  This is a utility function just for reducing code duplication.

  \param
    :log_likelihood_evaluator: evaluator object associated w/the state objects being constructed
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
  \output
    :state_vector[max_num_threads]: vector of states containing max_num_threads properly initialized state objects
\endrst*/
template <typename LogLikelihoodEvaluator>
OL_NONNULL_POINTERS void SetupLogLikelihoodState(const LogLikelihoodEvaluator& log_likelihood_evaluator,
                                                 const CovarianceInterface& covariance, int max_num_threads,
                                                 std::vector<typename LogLikelihoodEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(log_likelihood_evaluator, covariance);
  }
}

/*!\rst
  Select a valid (point, value) pair to represent the current best known objective value.

  This is a utility function just for reducing code duplication.

  \param
    :log_likelihood_evaluator: object supporting evaluation of log likelihood
    :initial_guesses[num_hyperparameters][num_multistarts]: list of hyperparameters at which to compute log likelihood
    :num_hyperparameters: dimension of the domain
    :num_multistarts: number of random points to draw
    :log_likelihood_state[1]: properly constructed/configured LogLikelihoodEvaluator::State object
    :io_container[1]: properly constructed OptimizationIOContainer object
  \output
    :log_likelihood_state[1]: internal states of state object may be modified
    :io_container[1]: OptimizationIOContainer with its best_objective_value and best_point fields set (according to check_all_points flag)
\endrst*/
template <typename LogLikelihoodEvaluator>
OL_NONNULL_POINTERS void InitializeBestKnownPoint(const LogLikelihoodEvaluator& log_likelihood_evaluator,
                                                  double const * restrict initial_guesses,
                                                  int num_hyperparameters, int num_multistarts,
                                                  typename LogLikelihoodEvaluator::StateType * log_likelihood_state,
                                                  OptimizationIOContainer * io_container) {
  // initialize io_container to the first point (arbitrary, but valid choice)
  io_container->best_objective_value_so_far = -std::numeric_limits<double>::infinity();
  std::copy(initial_guesses, initial_guesses + num_hyperparameters, io_container->best_point.data());

  // eval objective at all initial_guesses, set io_container to the best values
  for (int i = 0; i < num_multistarts; ++i) {
    log_likelihood_state->SetCurrentPoint(log_likelihood_evaluator, initial_guesses + i*num_hyperparameters);
    double log_likelihood = log_likelihood_evaluator.ComputeObjectiveFunction(log_likelihood_state);
    if (io_container->best_objective_value_so_far < log_likelihood) {
      io_container->best_objective_value_so_far = log_likelihood;
      std::copy(initial_guesses + i*num_hyperparameters, initial_guesses + (i+1)*num_hyperparameters,
                io_container->best_point.data());
    }
  }
}

/*!\rst
  Optimize a log likelihood measure of model fit (as a function of the hyperparameters
  of a covariance function) using the prior (i.e., sampled points, values).  Optimization is done
  using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from gpp_optimization.hpp.
  Please see that file for details on gradient descent and see gpp_optimizer_parameters.hpp for the meanings of
  the GradientDescentParameters.

  This function is just a simple wrapper that sets up the Evaluator's State and calls a general template for restarted GD.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline implying
  that this function should be backed by multistarting on a grid (or similar) to provide better chances of a good initial guess.

  The 'dumb' search component is provided through MultistartGradientDescentHyperparameterOptimization<...>(...) (see below).
  Generally, calling that function should be preferred.  This function is meant for:

  1. easier testing
  2. if you really know what you're doing

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  Let ``n_hyper = covariance_ptr->GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of log likelihood and its gradient
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
      covariance_ptr->GetCurrentHyperparameters() will be used to obtain the initial guess
    :gd_parameters: GradientDescentParameters object that describes the parameters controlling hyperparameter optimization (e.g., number
      of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
  \output
    :next_hyperparameters[n_hyper]: the new hyperparameters found by gradient descent
\endrst*/
template <typename LogLikelihoodEvaluator, typename DomainType>
OL_NONNULL_POINTERS void RestartedGradientDescentHyperparameterOptimization(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    const GradientDescentParameters& gd_parameters,
    const DomainType& domain,
    double * restrict next_hyperparameters) {
  if (unlikely(gd_parameters.max_num_restarts <= 0)) {
    return;
  }

  OL_VERBOSE_PRINTF("Hyperparameter Optimization via %s\n", OL_CURRENT_FUNCTION_NAME);
  typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_evaluator, covariance);

  GradientDescentOptimizer<LogLikelihoodEvaluator, DomainType> gd_opt;
  gd_opt.Optimize(log_likelihood_evaluator, gd_parameters, domain, &log_likelihood_state);
  log_likelihood_state.GetCurrentPoint(next_hyperparameters);
}

/*!\rst
  Function to add multistarting on top of (restarted) gradient descent hyperparameter optimization.
  Generates ``num_multistarts`` initial guesses (random sampling from domain), all within the specified domain, and kicks off
  an optimization run from each guess.

  Same idea as ComputeOptimalPointsToSampleWithRandomStarts() in gpp_math.hpp, which is for optimizing Expected Improvement;
  see those docs for additional gradient descent details.
  This is the primary endpoint for hyperparameter optimization using gradient descent.
  It constructs the required state objects, builds a GradientDescentOptimizer object, and wraps a series of calls:

  * The heart of multistarting is in MultistartOptimizer<...>::MultistartOptimize(...) (in gpp_optimization.hpp).

    * The heart of restarted GD is in GradientDescentOptimizer<...>::Optimize() (in gpp_optimization.hpp).
    * Log likelihood is computed in ComputeLogLikelihood() and its gradient in ComputeGradLogLikelihood(), which must be member
      functions of the LogLikelihoodEvaluator template parameter.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and gd_parameters.num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  .. Note:: the domain here must be specified in LOG-10 SPACE!

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING:: this function fails if NO improvement can be found!  In that case,
    ``best_next_point`` will always be the first randomly chosen point.
    ``found_flag`` will be set to false in this case.

  .. Note:: the domain here must be specified in LOG-10 SPACE!

  Let ``n_hyper = covariance_ptr->GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of gradient + hessian of log likelihood
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :gd_parameters: GradientDescentParameters object that describes the parameters controlling hyperparameter optimization (e.g., number
      of iterations, tolerances, learning rate)
    :domain[n_hyper]: array of ClosedInterval specifying the boundaries of a n_hyper-dimensional tensor-product domain.
      Specify in LOG-10 SPACE!
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if next_hyperparameters corresponds to a converged solution
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :next_hyperparameters[n_hyper]: the new hyperparameters found by gradient descent
\endrst*/
template <typename LogLikelihoodEvaluator>
OL_NONNULL_POINTERS void MultistartGradientDescentHyperparameterOptimization(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    const GradientDescentParameters& gd_parameters,
    ClosedInterval const * restrict domain,
    const ThreadSchedule& thread_schedule,
    bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator,
    double * restrict next_hyperparameters) {
  if (unlikely(gd_parameters.num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", gd_parameters.num_multistarts, 1);
  }

  const int num_hyperparameters = covariance.GetNumberOfHyperparameters();
  std::vector<double> initial_guesses(num_hyperparameters*gd_parameters.num_multistarts);
  std::vector<ClosedInterval> domain_linearspace_bounds(domain, domain + num_hyperparameters);
  ConvertFromLogToLinearDomainAndBuildInitialGuesses(num_hyperparameters, gd_parameters.num_multistarts,
                                                     uniform_generator, &domain_linearspace_bounds, &initial_guesses);

  TensorProductDomain domain_linearspace(domain_linearspace_bounds.data(), num_hyperparameters);

  // we need 1 state object per thread
  std::vector<typename LogLikelihoodEvaluator::StateType> log_likelihood_state_vector;
  SetupLogLikelihoodState(log_likelihood_evaluator, covariance, thread_schedule.max_num_threads,
                          &log_likelihood_state_vector);

  OptimizationIOContainer io_container(log_likelihood_state_vector[0].GetProblemSize());
  InitializeBestKnownPoint(log_likelihood_evaluator, initial_guesses.data(), num_hyperparameters,
                           gd_parameters.num_multistarts, log_likelihood_state_vector.data(), &io_container);

  GradientDescentOptimizer<LogLikelihoodEvaluator, TensorProductDomain> gd_opt;
  MultistartOptimizer<GradientDescentOptimizer<LogLikelihoodEvaluator, TensorProductDomain> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(gd_opt, log_likelihood_evaluator, gd_parameters,
                                          domain_linearspace, thread_schedule,
                                          initial_guesses.data(), gd_parameters.num_multistarts,
                                          log_likelihood_state_vector.data(),
                                          nullptr, &io_container);

  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), next_hyperparameters);
}

/*!\rst
  Optimize a log likelihood measure of model fit (as a function of the hyperparameters
  of a covariance function) using the prior (i.e., sampled points, values).  Optimization is done
  using Newton's method for optimization, via NewtonOptimization() from gpp_optimization.hpp.
  Please see that file for details on Newton and see gpp_optimizer_parameters.hpp for the meanings of the NewtonParameters.

  This function is just a simple wrapper that sets up the Evaluator's State and calls a general template for Newton,
  NewtonOptimization<...>(...) (in gpp_optimization.hpp).

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline implying
  that this function should be backed by multistarting on a grid (or similar) to provide better chances of a good initial guess.

  The 'dumb' search component is provided through MultistartNewtonHyperparameterOptimization<...>(...) (see below).
  Generally, calling that function should be preferred.  This is meant for:

  1. easier testing
  2. if you really know what you're doing

  | ``gamma = 1.01, time_factor = 1.0e-3`` should lead to good robustness at reasonable speed.  This should be a fairly safe default.
  | ``gamma = 1.05, time_factor = 1.0e-1`` will be several times faster but not as robust.

  Let ``n_hyper = covariance.GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of gradient + hessian of log likelihood
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
      covariance.GetCurrentHyperparameters() will be used to obtain the initial guess
    :newton_parameters: NewtonParameters object that describes the parameters controlling hyperparameter optimization (e.g., number
      of iterations, tolerances, diagonal dominance)
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
  \output
    :next_hyperparameters[n_hyper]: the new hyperparameters found by newton
\endrst*/
template <typename LogLikelihoodEvaluator, typename DomainType>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int NewtonHyperparameterOptimization(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    const NewtonParameters& newton_parameters,
    const DomainType& domain,
    double * restrict next_hyperparameters) {
  if (unlikely(newton_parameters.max_num_restarts <= 0)) {
    return 0;
  }

  OL_VERBOSE_PRINTF("Hyperparameter Optimization via %s\n", OL_CURRENT_FUNCTION_NAME);
  typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_evaluator, covariance);

  NewtonOptimizer<LogLikelihoodEvaluator, DomainType> newton_opt;
  int errors = newton_opt.Optimize(log_likelihood_evaluator, newton_parameters, domain, &log_likelihood_state);
  log_likelihood_state.GetCurrentPoint(next_hyperparameters);
  return errors;
}

/*!\rst
  Function to add multistarting on top of newton hyperparameter optimization.
  Generates ``num_multistarts`` initial guesses (random sampling from domain), all within the specified domain, and kicks off
  an optimization run from each guess.

  Same idea as ComputeOptimalPointsToSampleWithRandomStarts() in gpp_math.hpp, which is for optimizing Expected Improvement.
  This is the primary endpoint for hyperparameter optimization using Newton's method.
  It constructs the required state objects, builds a NewtonOptimizer object, and wraps a series of calls:

  * The heart of multistarting is in MultistartOptimizer<...>::MultistartOptimize<...>(...) (in gpp_optimization.hpp).

    * The heart of Newton is in NewtonOptimization() (in gpp_optimization.hpp).
    * Log likelihood is computed in ComputeLogLikelihood(), its gradient in ComputeGradLogLikelihood(), and its hessian in
    * ComputeHessianLogLikelihood(), which must be member functions of the LogLikelihoodEvaluator template parameter.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and gd_parameters.num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING:: this function fails if NO improvement can be found!  In that case,
    ``best_next_point`` will always be the first randomly chosen point.
    ``found_flag`` will be set to false in this case.

  .. Note:: the domain here must be specified in LOG-10 SPACE!

  Let ``n_hyper = covariance.GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of gradient + hessian of log likelihood
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :newton_parameters: NewtonParameters object that describes the parameters controlling hyperparameter optimization (e.g., number
      of iterations, tolerances, diagonal dominance)
    :domain[n_hyper]: array of ClosedInterval specifying the boundaries of a n_hyper-dimensional tensor-product domain.
      Specify in LOG-10 SPACE!
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if next_hyperparameters corresponds to a converged solution
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :next_hyperparameters[n_hyper]: the new hyperparameters found by newton
\endrst*/
template <typename LogLikelihoodEvaluator>
OL_NONNULL_POINTERS void MultistartNewtonHyperparameterOptimization(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    const NewtonParameters& newton_parameters,
    ClosedInterval const * restrict domain,
    const ThreadSchedule& thread_schedule,
    bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator,
    double * restrict next_hyperparameters) {
  if (unlikely(newton_parameters.num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", newton_parameters.num_multistarts, 1);
  }

  const int num_hyperparameters = covariance.GetNumberOfHyperparameters();
  std::vector<double> initial_guesses(num_hyperparameters*newton_parameters.num_multistarts);
  std::vector<ClosedInterval> domain_linearspace_bounds(domain, domain + num_hyperparameters);
  ConvertFromLogToLinearDomainAndBuildInitialGuesses(num_hyperparameters, newton_parameters.num_multistarts,
                                                     uniform_generator, &domain_linearspace_bounds, &initial_guesses);

  TensorProductDomain domain_linearspace(domain_linearspace_bounds.data(), num_hyperparameters);

  // we need 1 state object per thread
  std::vector<typename LogLikelihoodEvaluator::StateType> log_likelihood_state_vector;
  SetupLogLikelihoodState(log_likelihood_evaluator, covariance, thread_schedule.max_num_threads,
                          &log_likelihood_state_vector);

  OptimizationIOContainer io_container(log_likelihood_state_vector[0].GetProblemSize());
  InitializeBestKnownPoint(log_likelihood_evaluator, initial_guesses.data(), num_hyperparameters,
                           newton_parameters.num_multistarts, log_likelihood_state_vector.data(), &io_container);

  NewtonOptimizer<LogLikelihoodEvaluator, TensorProductDomain> newton_opt;
  MultistartOptimizer<NewtonOptimizer<LogLikelihoodEvaluator, TensorProductDomain> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(newton_opt, log_likelihood_evaluator, newton_parameters,
                                          domain_linearspace, thread_schedule, initial_guesses.data(),
                                          newton_parameters.num_multistarts,
                                          log_likelihood_state_vector.data(),
                                          nullptr, &io_container);

  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), next_hyperparameters);
}

/*!\rst
  Function to evaluate various log likelihood measures over a specified list of num_multistarts hyperparameters.
  Optionally outputs the log likelihood at each of these hyperparameters.
  Outputs the hyperparameters of the input set obtaining the maximum log likelihood value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
  This function is also useful for plotting or debugging purposes (just to get a bunch of log likelihood values).

  This function is just a wrapper that builds the required state objects and a NullOptimizer object and calls
  MultistartOptimizer<...>::MultistartOptimize(...); see gpp_optimization.hpp.

  Let ``n_hyper = covariance.GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of log likelihood
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp), specify in LINEAR SPACE!
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_guided), chunk_size (0).
    :initial_guesses[n_hyper][num_multistarts]: list of hyperparameters at which to compute log likelihood
    :num_multistarts: number of random points to generate for use as initial guesses
  \output
    :found_flag[1]: true if next_hyperparameters corresponds to a finite log likelihood
    :function_values[num_multistarts]: log likelihood evaluated at each point of initial_guesses, in the same order as initial_guesses; never dereferenced if nullptr
    :next_hyperparameters[n_hyper]: the new hyperparameters found by "dumb" search
\endrst*/
template <typename LogLikelihoodEvaluator, typename DomainType>
void EvaluateLogLikelihoodAtPointList(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    const DomainType& domain_linearspace,
    const ThreadSchedule& thread_schedule,
    double const * restrict initial_guesses,
    int num_multistarts,
    bool * restrict found_flag,
    double * restrict function_values,
    double * restrict next_hyperparameters) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  std::vector<typename LogLikelihoodEvaluator::StateType> log_likelihood_state_vector;
  SetupLogLikelihoodState(log_likelihood_evaluator, covariance, thread_schedule.max_num_threads,
                          &log_likelihood_state_vector);

  // initialize io_container to the first point (arbitrary, but valid choice)
  // set the value to -infinity
  OptimizationIOContainer io_container(log_likelihood_state_vector[0].GetProblemSize(),
                                       -std::numeric_limits<double>::infinity(),
                                       initial_guesses);

  NullOptimizer<LogLikelihoodEvaluator, DomainType> null_opt;
  typename NullOptimizer<LogLikelihoodEvaluator, DomainType>::ParameterStruct null_parameters;
  MultistartOptimizer<NullOptimizer<LogLikelihoodEvaluator, DomainType> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(null_opt, log_likelihood_evaluator, null_parameters,
                                          domain_linearspace, thread_schedule, initial_guesses,
                                          num_multistarts, log_likelihood_state_vector.data(),
                                          function_values, &io_container);

  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), next_hyperparameters);
}

/*!\rst
  Function to do a "dumb" search over num_multistarts points (generated on a Latin Hypercube) for the optimal set of
  hyperparameters (largest log likelihood).

  Generally gradient descent or newton are preferred but when they fail to converge this may be the only "robust" option.

  This function wraps EvaluateLogLikelihoodAtPointList(), providing it with a uniformly sampled (on a latin hypercube) set of
  hyperparameters at which to evaluate log likelihood.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. Note:: the domain here must be specified in LOG-10 SPACE!

  Let ``n_hyper = covariance.GetNumberOfHyperparameters();``

  \param
    :log_likelihood_evaluator: object supporting evaluation of log likelihood
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :domain[n_hyper]: array of ClosedInterval specifying the boundaries of a n_hyper-dimensional tensor-product domain.
      Specify in LOG-10 SPACE!
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_guided), chunk_size (0).
    :num_multistarts: number of random points to generate for use as initial guesses
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :found_flag[1]: true if next_hyperparameters corresponds to a finite log likelihood
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :next_hyperparameters[n_hyper]: the new hyperparameters found by "dumb" search
\endrst*/
template <typename LogLikelihoodEvaluator>
OL_NONNULL_POINTERS void LatinHypercubeSearchHyperparameterOptimization(
    const LogLikelihoodEvaluator& log_likelihood_evaluator,
    const CovarianceInterface& covariance,
    ClosedInterval const * restrict domain,
    const ThreadSchedule& thread_schedule,
    int num_multistarts,
    bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator,
    double * restrict next_hyperparameters) {
  const int num_hyperparameters = covariance.GetNumberOfHyperparameters();
  std::vector<double> initial_guesses(num_hyperparameters*num_multistarts);
  std::vector<ClosedInterval> domain_linearspace_bounds(domain, domain + num_hyperparameters);
  ConvertFromLogToLinearDomainAndBuildInitialGuesses(num_hyperparameters, num_multistarts, uniform_generator,
                                                     &domain_linearspace_bounds, &initial_guesses);

  TensorProductDomain domain_linearspace(domain_linearspace_bounds.data(), num_hyperparameters);

  EvaluateLogLikelihoodAtPointList(log_likelihood_evaluator, covariance, domain_linearspace,
                                   thread_schedule, initial_guesses.data(), num_multistarts,
                                   found_flag, nullptr, next_hyperparameters);
}

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_MODEL_SELECTION_HPP_
