/*!
  \file gpp_math.hpp
  \rst
  1. OVERVIEW OF GAUSSIAN PROCESSES AND EXPECTED IMPROVEMENT; WHAT ARE WE TRYING TO DO?
  2. FILE OVERVIEW
  3. IMPLEMENTATION NOTES
  4. NOTATION
  5. CITATIONS

  **1. OVERVIEW OF GAUSSIAN PROCESSES AND EXPECTED IMPROVEMENT; WHAT ARE WE TRYING TO DO?**

  .. Note:: these comments are copied in Python: interfaces/__init__.py

  At a high level, this file optimizes an objective function \ms f(x)\me.  This operation
  requires data/uncertainties about prior and concurrent experiments as well as
  a covariance function describing how these data [are expected to] relate to each
  other.  The points x represent experiments. If \ms f(x)\me is say, survival rate for
  a drug test, the dimensions of x might include dosage amount, dosage frequency,
  and overall drug-use time-span.

  The objective function is not required in closed form; instead, only the ability
  to sample it at points of interest is needed.  Thus, the optimization process
  cannot work with \ms f(x)\me directly; instead a surrogate is built via interpolation
  with Gaussian Proccesses (GPs).

  Following Rasmussen & Williams (2.2), a Gaussian Process is a collection of random
  variables, any finite number of which have a joint Gaussian distribution (Defn 2.1).
  Hence a GP is fully specified by its mean function, \ms m(x)\me, and covariance function,
  \ms k(x,x')\me.  Then we assume that a real process \ms f(x)\me (e.g., drug survival rate) is
  distributed like:

  .. math:: f(x) ~ GP(m(x), k(x,x'))

  with

  .. math:: m(x) = E[f(x)], k(x,x') = E[(f(x) - m(x))*(f(x') - m(x'))].

  Then sampling from \ms f(x)\me is simply drawing from a Gaussian with the appropriate mean
  and variance.

  However, since we do not know \ms f(x)\me, we cannot precisely build its corresponding GP.
  Instead, using samples from \ms f(x)\me (e.g., by measuring experimental outcomes), we can
  iteratively improve our estimate of \ms f(x)\me.  See GaussianProcess class docs
  and implementation docs for details on how this is done.

  The optimization process models the objective using a Gaussian process (GP) prior
  (also called a GP predictor) based on the specified covariance and the input
  data (e.g., through member functions ComputeMeanOfPoints, ComputeVarianceOfPoints).  Using the GP,
  we can compute the expected improvement (EI) from sampling any particular point.  EI
  is defined relative to the best currently known value, and it represents what the
  algorithm believes is the most likely outcome from sampling a particular point in parameter
  space (aka conducting a particular experiment).

  See ExpectedImprovementEvaluator and OnePotentialSampleExpectedImprovementEvaluator class
  docs for further details on computing EI.  Both support ComputeExpectedImprovement() and
  ComputeGradExpectedImprovement().

  The dimension of the GP is equal to the number of simultaneous experiments being run;
  i.e., the GP may be multivariate.  The behavior of the GP is controlled by its underlying
  covariance function and the data/uncertainty of prior points (experiments).

  With the ability the compute EI, the final step is to optimize
  to find the best EI.  This is done using multistart gradient descent (MGD), in
  ComputeOptimalPointsToSample(). This method wraps a MGD call and falls back on random search
  if that fails. See gpp_optimization.hpp for multistart/optimization templates. This method
  can evaluate and optimize EI at serval points simultaneously; e.g., if we wanted to run 4 simultaneous
  experiments, we can use EI to select all 4 points at once.

  The literature (e.g., Ginsbourger 2008) refers to these problems collectively as q-EI, where q
  is a positive integer. So 1-EI is the originally dicussed usage, and the previous scenario with
  multiple simultaneous points/experiments would be called 4-EI.

  Additionally, there are use cases where we have existing experiments that are not yet complete but
  we have an opportunity to start some new trials. For example, maybe we are a drug company currently
  testing 2 combinations of dosage levels. We got some new funding, and can now afford to test
  3 more sets of dosage parameters. Ideally, the decision on the new experiments should depend on
  the existence of the 2 ongoing tests. We may not have any data from the ongoing experiments yet;
  e.g., they are [double]-blind trials. If nothing else, we would not want to duplicate any
  existing experiments! So we want to solve 3-EI using the knowledge of the 2 ongoing experiments.

  We call this q,p-EI, so the previous example would be 3,2-EI. The q-EI notation is equivalent to
  q,0-EI; if we do not explicitly write the value of p, it is 0. So q is the number of new
  (simultaneous) experiments to select. In code, this would be the size of the output from EI
  optimization (i.e., ``best_points_to_sample``, of which there are ``q = num_to_sample points``).
  p is the number of ongoing/incomplete experiments to take into account (i.e., ``points_being_sampled``
  of which there are ``p = num_being_sampled`` points).

  Back to optimization: the idea behind gradient descent is simple.  The gradient gives us the
  direction of steepest ascent (negative gradient is steepest descent).  So each iteration, we
  compute the gradient and take a step in that direction.  The size of the step is not specified
  by GD and is left to the specific implementation.  Basically if we take steps that are
  too large, we run the risk of over-shooting the solution and even diverging.  If we
  take steps that are too small, it may take an intractably long time to reach the solution.
  Thus the magic is in choosing the step size; we do not claim that our implementation is
  perfect, but it seems to work reasonably.  See ``gpp_optimization.hpp`` for more details about
  GD as well as the template definition.

  For particularly difficult problems or problems where gradient descent's parameters are not
  well-chosen, GD can fail to converge.  If this happens, we can fall back on heuristics;
  e.g., 'dumb' search (i.e., evaluate EI at a large number of random points and take the best
  one). Naive search lives in: ComputeOptimalPointsToSampleViaLatinHypercubeSearch<>().

  **2. FILE OVERVIEW**

  This file contains mathematical functions supporting optimal learning.
  These include functions to compute characteristics of Gaussian Processes
  (e.g., variance, mean) and the gradients of these quantities as well as functions to
  compute and optimize the expected improvement.

  Functions here generally require some combination of a CovarianceInterface object as well as
  data about prior and current (i.e., concurrent) experiments.  These data are encapsulated in
  the GaussianProcess class.  Then we build an ExpectedImprovementEvaluator object (with
  associated state, see ``gpp_common.hpp`` item 5 for (Evaluator, State) relations) on top of a
  GaussianProcess for computing and optimizing EI.

  For further theoretical details about Gaussian Processes, see
  Rasmussen and Williams, Gaussian Processes for Machine Learning (2006).
  A bare-bones summary is provided in ``gpp_math.cpp``.

  For further details about expected improvement and the optimization thereof,
  see Scott Clark's PhD thesis.  Again, a summary is provided in ``gpp_math.cpp``'s file comments.

  **3. IMPLEMENTATION NOTES**

  a. This file has a few primary endpoints for EI optimization:

     i. ComputeOptimalPointsToSampleWithRandomStarts<>():

        Solves the q,p-EI problem.

        Takes in a gaussian_process describing the prior, domain, config, etc.; outputs the next best point(s) (experiment)
        to sample (run). Uses gradient descent.

     ii. ComputeOptimalPointsToSampleViaLatinHypercubeSearch<>():

         Estimates the q,p-EI problem.

         Takes in a gaussian_process describing the prior, domain, etc.; outputs the next best point(s) (experiment)
         to sample (run). Uses 'dumb' search.

     iii. ComputeOptimalPointsToSample<>() (Recommended):

          Solves the q,p-EI problem.

          Wraps the previous two items; relies on gradient descent and falls back to "dumb" search if it fails.

     .. NOTE::
         See ``gpp_math.cpp``'s header comments for more detailed implementation notes.

         There are also several other functions with external linkage in this header; these
         are provided primarily to ease testing and to permit lower level access from python.

  b. See ``gpp_common.hpp`` header comments for additional implementation notes.

  **4. NOTATION**

  And domain-specific notation, following Rasmussen, Williams:

    * ``X = points_sampled``; this is the training data (size ``dim`` X ``num_sampled``), also called the design matrix
    * ``Xs = points_to_sample``; this is the test data (size ``dim`` X num_to_sample``)
    * ``y, f, f(x) = points_sampled_value``, the experimental results from sampling training points
    * ``K, K_{ij}, K(X,X) = covariance(X_i, X_j)``, covariance matrix between training inputs (``num_sampled x num_sampled``)
    * ``Ks, Ks_{ij}, K(X,Xs) = covariance(X_i, Xs_j)``, covariance matrix between training and test inputs (``num_sampled x num_to_sample``)
    * ``Kss, Kss_{ij}, K(Xs,Xs) = covariance(Xs_i, Xs_j)``, covariance matrix between test inputs (``num_to_sample x num_to_sample``)
    * ``\theta``: (vector) of hyperparameters for a covariance function

  .. NOTE::
       Due to confusion with multiplication (K_* looks awkward in code comments), Rasmussen & Williams' \ms K_*\me
       notation has been repalced with ``Ks`` and \ms K_{**}\me is ``Kss``.

  Connecting to the q,p-EI notation, both the points represented by "q" and "p" are represented by ``Xs``. Within
  the GP, there is no distinction between points being sampled by ongoing experiments and new points to sample.

  **5. CITATIONS**

  a. Gaussian Processes for Machine Learning.
     Carl Edward Rasmussen and Christopher K. I. Williams. 2006.
     Massachusetts Institute of Technology.  55 Hayward St., Cambridge, MA 02142.
     http://www.gaussianprocess.org/gpml/ (free electronic copy)

  b. Parallel Machine Learning Algorithms In Bioinformatics and Global Optimization (PhD Dissertation).
     Part II, EPI: Expected Parallel Improvement
     Scott Clark. 2012.
     Cornell University, Center for Applied Mathematics.  Ithaca, NY.
     https://github.com/sc932/Thesis
     sclark@yelp.com

  c. Differentiation of the Cholesky Algorithm.
     S. P. Smith. 1995.
     Journal of Computational and Graphical Statistics. Volume 4. Number 2. p134-147

  d. A Multi-points Criterion for Deterministic Parallel Global Optimization based on Gaussian Processes.
     David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.  2008.
     D´epartement 3MI. Ecole Nationale Sup´erieure des Mines. 158 cours Fauriel, Saint-Etienne, France.
     ginsbourger@emse.fr, leriche@emse.fr, carraro@emse.fr

  e. Efficient Global Optimization of Expensive Black-Box Functions.
     Jones, D.R., Schonlau, M., Welch, W.J. 1998.
     Journal of Global Optimization, 13, 455-492.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_covariance.hpp"
#include "gpp_logging.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

struct ThreadSchedule;
struct PointsToSampleState;

/*!\rst
  Object that encapsulates Gaussian Process Priors (GPPs).  A GPP is defined by a set of
  (sample point, function value, noise variance) triples along with a covariance function that relates the points.
  Each point has dimension dim.  These are the training data; for example, each sample point might specify an experimental
  cohort and the corresponding function value is the objective measured for that experiment.  There is one noise variance
  value per function value; this is the measurement error and is treated as N(0, noise_variance) Gaussian noise.

  GPPs estimate a real process \ms f(x) = GP(m(x), k(x,x'))\me (see file docs).  This class deals with building an estimator
  to the actual process using measurements taken from the actual process--the (sample point, function val, noise) triple.
  Then predictions about unknown points can be made by sampling from the GPP--in particular, finding the (predicted)
  mean and variance.  These functions (and their gradients) are provided in ComputeMeanOfPoints, ComputeVarianceOfPoints,
  etc.

  Further mathematical details are given in the implementation comments, but we are essentially computing:

  | ComputeMeanOfPoints    : ``K(Xs, X) * [K(X,X) + \sigma_n^2 I]^{-1} * y``
  | ComputeVarianceOfPoints: ``K(Xs, Xs) - K(Xs,X) * [K(X,X) + \sigma_n^2 I]^{-1} * K(X,Xs)``

  This (estimated) mean and variance characterize the predicted distributions of the actual \ms m(x), k(x,x')\me
  functions that underly our GP.

  .. Note:: the preceding comments are copied in Python: interfaces/gaussian_process_interface.py

  For testing and experimental purposes, this class provides a framework for sampling points from the GP (i.e., given a
  point to sample and predicted measurement noise) as well as adding additional points to an already-formed GP.  Sampling
  points requires drawing from \ms N(0,1)\me so this class also holds PRNG state to do so via the NormalRNG object from gpp_random.

  .. NOTE::
       Functions that manipulate the PRNG directly or indirectly (changing state, generating points)
       are NOT THREAD-SAFE. All thread-safe functions are marked const.

  These mean/variance methods require some external state: namely, the set of potential points to sample.  Additionally,
  temporaries and derived quantities depending on these "points to sample" eliminate redundant computation.  This external
  state is handled through PointsToSampleState objects, which are constructed separately and filled through
  PointsToSampleState::SetupState() which interacts with functions in this class.
\endrst*/
class GaussianProcess final {
 public:
  using StateType = PointsToSampleState;
  using NormalGeneratorType = NormalRNG;
  using EngineType = NormalGeneratorType::EngineType;

  //! Default seed value to make reproducing test results simple.
  static constexpr EngineType::result_type kDefaultSeed = 87214;

  //! Minimum allowed standard deviation value in ComputeGradCholeskyVarianceOfPointsPerPoint (= machine precision).
  //! Values that are too small result in problems b/c we may compute ``std_dev/var`` (which is enormous
  //! if ``std_dev = 1.0e-150`` and ``var = 1.0e-300``) since this only arises when we fail to compute ``std_dev = var = 0.0``.
  //! Note: this is only relevant if noise = 0.0; this minimum will not affect GPs with noise since this value
  //! is below the smallest amount of noise users can meaningfully add.
  //! This value was chosen to be consistent with the singularity condition in CholeskyFactorL
  //! and tested for robustness with the setup in EIOnePotentialSampleEdgeCasesTest().
  static constexpr double kMinimumStdDev = std::numeric_limits<double>::epsilon();

  /*!\rst
    Constructs a GaussianProcess object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    .. Warning::
        ``points_sampled`` is not allowed to contain duplicate points; doing so results in singular covariance matrices.

    \param
      :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
      :points_sampled[dim][num_sampled]: points that have already been sampled
      :points_sampled_value[num_sampled]: values of the already-sampled points
      :noise_variance[num_sampled]: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_sampled: number of already-sampled points
  \endrst*/
  GaussianProcess(const CovarianceInterface& covariance_in,
                  double const * restrict points_sampled_in,
                  double const * restrict points_sampled_value_in,
                  double const * restrict noise_variance_in,
                  int dim_in, int num_sampled_in) OL_NONNULL_POINTERS;

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_sampled() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_sampled_;
  }

  const std::vector<double>& points_sampled() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return points_sampled_;
  }

  const std::vector<double>& points_sampled_value() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return points_sampled_value_;
  }

  const std::vector<double>& noise_variance() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return noise_variance_;
  }

  /*!\rst
    Change the hyperparameters of this GP's covariance function.
    Also forces recomputation of all derived quantities for GP to remain consistent.

    .. WARNING::
         Using this function invalidates any PointsToSampleState objects created with "this" object.
         For any such objects "state", call state.SetupState(...) to restore them.

    \param
      :hyperparameters_new[covariance_ptr->GetNumberOfHyperparameters]: new hyperparameter array
  \endrst*/
  void SetCovarianceHyperparameters(double const * restrict hyperparameters_new) OL_NONNULL_POINTERS {
    covariance_ptr_->SetHyperparameters(hyperparameters_new);
    RecomputeDerivedVariables();
  }

  /*!\rst
    Sets up the PointsToSampleState object so that it can be used to compute GP mean, variance, and gradients thereof.
    ASSUMES all needed space is ALREADY ALLOCATED.

    This function should not be called directly; instead use PointsToSampleState::SetupState().

    \param
      :points_to_sample_state[1]: pointer to a PointsToSampleState object where all space has been properly allocated
    \output
      :points_to_sample_state[1]: pointer to a fully configured PointsToSampleState object. overwrites input
  \endrst*/
  void FillPointsToSampleState(StateType * points_to_sample_state) const OL_NONNULL_POINTERS;

  /*!\rst
    Add the specified (point, fcn value, noise variance) historical data to this GP.

    Forces recomputation of all derived quantities for GP to remain consistent.

    \param
      :new_points[dim][num_new_points]: coordinates of each new point to add
      :new_points_value[num_new_points]: function value at each new point
      :new_points_noise_variance[num_new_points]: \sigma_n^2 corresponding to the signal noise in measuring new_points_value
      :num_new_points: number of new points to add to the GP
  \endrst*/
  void AddPointsToGP(double const * restrict new_points,
                     double const * restrict new_points_value,
                     double const * restrict new_points_noise_variance,
                     int num_new_points);

  /*!\rst
    Sample a function value from a Gaussian Process prior, provided a point at which to sample.

    Uses the formula ``function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2``, where ``w1, w2``
    are draws from \ms N(0,1)\me.

    .. NOTE::
         Set noise_variance to 0 if you want "accurate" draws from the GP.
         BUT if the drawn (point, value) pair is meant to be added back into the GP (e.g., for testing), then this point
         MUST be drawn with noise_variance equal to the noise associated with "point" as a member of "points_sampled"

    \param
      :point_to_sample[dim]: coordinates of the point at which to generate a function value (from GP)
      :noise_variance_this_point: if this point is to be added into the GP, it needs to be generated with its associated noise var
    \return
      function value drawn from this GP
  \endrst*/
  double SamplePointFromGP(double const * restrict point_to_sample,
                           double noise_variance_this_point) noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Computes the mean of this GP at each of ``Xs`` (``points_to_sample``).

    .. Note:: ``points_to_sample`` should not contain duplicate points.

    .. Note:: comments are copied in Python: interfaces/gaussian_process_interface.py

    \param
      :points_to_sample_state: a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
    \output
      :mean_of_points[num_to_sample]: mean of GP, one per GP dimension
  \endrst*/
  void ComputeMeanOfPoints(const StateType& points_to_sample_state,
                           double * restrict mean_of_points) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Computes the gradient of the mean of this GP at each of ``Xs`` (``points_to_sample``) wrt ``Xs``.

    .. Note:: ``points_to_sample`` should not contain duplicate points.

    Note that ``grad_mu`` is nominally sized: ``grad_mu[dim][num_to_sample][num_to_sample]``.
    However, for ``0 <= i,j < num_to_sample``, ``i != j``, ``grad_mu[d][i][j] = 0``.
    (See references or implementation for further details.)
    Thus, ``grad_mu`` is stored in a reduced form which only tracks the nonzero entries.

    .. Note:: comments are copied in Python: interfaces/gaussian_process_interface.py

    \param
      :points_to_sample_state: a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
    \output
      :grad_mu[dim][state.num_derivatives]: gradient of the mean of the GP.  ``grad_mu[d][i]`` is
        actually the gradient of ``\mu_i`` with respect to ``x_{d,i}``, the d-th dimension of
        the i-th entry of ``points_to_sample``.
  \endrst*/
  void ComputeGradMeanOfPoints(const StateType& points_to_sample_state,
                               double * restrict grad_mu) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Computes the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

    The variance matrix is symmetric (in fact, SPD) and is stored in the LOWER TRIANGLE.

    .. Note:: ``points_to_sample`` should not contain duplicate points.

    .. Note:: comments are copied in Python: interfaces/gaussian_process_interface.py

    \param
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
    \output
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState; only temporary state may be mutated
      :var_star[num_to_sample][num_to_sample]: variance of GP evaluated at ``points_to_sample``
  \endrst*/
  void ComputeVarianceOfPoints(StateType * points_to_sample_state,
                               double * restrict var_star) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Similar to ComputeGradCholeskyVarianceOfPoints() except this does not include the gradient terms from
    the cholesky factorization.  Description will not be duplicated here.
  \endrst*/
  void ComputeGradVarianceOfPoints(StateType * points_to_sample_state,
                                   double * restrict grad_var) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Computes the gradient of the cholesky factorization of the variance of this GP with respect to ``points_to_sample``.
    This function accounts for the effect on the gradient resulting from
    cholesky-factoring the variance matrix.  See Smith 1995 for algorithm details.

    ``points_to_sample`` is not allowed to contain duplicate points. Violating this results in a singular variance matrix.

    Note that ``grad_chol`` is nominally sized:

    ``grad_chol[dim][num_to_sample][num_to_sample][num_to_sample]``.

    Let this be indexed ``grad_chol[d][i][j][k]``, which is read the derivative of ``var[i][j]``
    with respect to ``x_{d,k}`` (x = ``points_to_sample``)

    .. Note:: comments are copied in Python: interfaces/gaussian_process_interface.py

    \param
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
      :chol_var[num_to_sample][num_to_sample]: the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``)
        e.g., from the cholesky factorization of ``ComputeVarianceOfPoints``
    \output
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState; only temporary state may be mutated
      :grad_chol[dim][num_to_sample][num_to_sample][state->num_derivatives]: gradient of the cholesky-factored
        variance of the GP.  ``grad_chol[d][i][j][k]`` is actually the gradients of ``var_{i,j}`` with
        respect to ``x_{d,k}``, the d-th dimension of the k-th entry of ``points_to_sample``
  \endrst*/
  void ComputeGradCholeskyVarianceOfPoints(StateType * points_to_sample_state,
                                           double const * restrict chol_var,
                                           double * restrict grad_chol) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Seed the random number generator with the specified seed.
    See gpp_random, struct NormalRNG for details.

    \param
      :seed: new seed to set
  \endrst*/
  void SetExplicitSeed(EngineType::result_type seed) noexcept;

  /*!\rst
    Seed the random number generator using a combination of the specified seed,
    current time, and potentially other factors.
    See gpp_random, struct NormalRNG for details.

    \param
      :seed: base value for new seed
  \endrst*/
  void SetRandomizedSeed(EngineType::result_type seed) noexcept;

  /*!\rst
    Seeds the generator with its last used seed value.
    Useful for testing--e.g., can conduct multiple runs with the same initial conditions
  \endrst*/
  void ResetToMostRecentSeed() noexcept;

  /*!\rst
    Clones "this" GaussianProcess.

    \return
      Pointer to a constructed object that is a copy of "this"
  \endrst*/
  GaussianProcess * Clone() const OL_WARN_UNUSED_RESULT;

  OL_DISALLOW_DEFAULT_AND_ASSIGN(GaussianProcess);

 protected:
  explicit GaussianProcess(const GaussianProcess& source);

 private:
  void BuildCovarianceMatrixWithNoiseVariance() noexcept;
  void BuildMixCovarianceMatrix(double const * restrict points_to_sample, int num_to_sample,
                                double * restrict cov_mat) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Similar to ComputeGradCholeskyVarianceOfPointsPerPoint() except this does not include the gradient terms from
    the cholesky factorization.  Description will not be duplicated here.
  \endrst*/
  void ComputeGradVarianceOfPointsPerPoint(StateType * points_to_sample_state, int diff_index,
                                           double * restrict grad_var) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Computes the gradient of the cholesky factorization of the variance of this GP with respect to the
    ``diff_index``-th point in ``points_to_sample``.

    This internal method is meant to be used by ComputeGradCholeskyVarianceOfPoints() to construct the gradient wrt all
    points of ``points_to_sample``. See that function for more details.

    \param
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
      :diff_index: index of ``points_to_sample`` in {0, .. ``num_to_sample``-1} to be differentiated against
      :chol_var[num_to_sample][num_to_sample]: the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``)
        e.g., from the cholesky factorization of ``ComputeVarianceOfPoints``
    \output
      :points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState; only temporary state may be mutated
      :grad_chol[dim][num_to_sample][num_to_sample]: gradient of the cholesky-factored
        variance of the GP.  ``grad_chol[d][i][j]`` is actually the gradients of ``var_{i,j}`` with
        respect to ``x_{d,k}``, the d-th dimension of the k-th entry of ``points_to_sample``, where
        k = ``diff_index``
  \endrst*/
  void ComputeGradCholeskyVarianceOfPointsPerPoint(StateType * points_to_sample_state, int diff_index,
                                                   double const * restrict chol_var,
                                                   double * restrict grad_chol) const noexcept OL_NONNULL_POINTERS;

  /*!\rst
    Recomputes (including resizing as needed) the derived quantities in this class.
    This function should be called any time state variables are changed.
  \endrst*/
  void RecomputeDerivedVariables();


  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points in ``points_sampled``
  int num_sampled_;

  // state variables for prior
  //! covariance class (for computing covariance and its gradients)
  std::unique_ptr<CovarianceInterface> covariance_ptr_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;

  // derived variables for prior
  //! cholesky factorization of ``K`` (i.e., ``K(X,X)`` covariance matrix (prior), includes noise variance)
  std::vector<double> K_chol_;
  //! ``K^-1 * y``; computed WITHOUT forming ``K^-1``
  std::vector<double> K_inv_y_;

  //! Normal PRNG for use with sampling points from GP
  NormalGeneratorType normal_rng_;
};

/*!\rst
  This object holds the state needed for a GaussianProcess object characterize the distribution of function values arising from
  sampling the GP at a list of ``points_to_sample``.  This object is required by the GaussianProcess to access functionality for
  computing the mean, variance, and spatial gradients thereof.

  The "independent variables" for this object are ``points_to_sample``. These points are both the "p" and the "q" in q,p-EI;
  i.e., they are the parameters of both ongoing experiments and new predictions. Recall that in q,p-EI, the q points are
  called ``points_to_sample`` and the p points are called ``points_being_sampled.`` Here, we need to make predictions about
  both point sets with the GP, so we simply call the union of point sets ``points_to_sample.``

  In GP computations, there is really no distinction between the "q" and "p" points from EI, ``points_to_sample`` and
  ``points_being_sampled``, respectively. However, in EI optimization, we only need gradients of GP quantities wrt
  ``points_to_sample``, so users should build PointsToSampleState() with ``num_derivatives = num_to_sample``.

  Once constructed, this object provides the SetupState() function to update it for computations at different sets of
  potential points to sample.

  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
struct PointsToSampleState final {
  /*!\rst
    Constructs a PointsToSampleState object with new ``points_to_sample``.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that GaussianProcess's mean, variance (and gradients thereof) functions can be called.

    .. WARNING::
         This object's state is INVALIDATED if the gaussian_process used in construction is mutated!
         SetupState() should be called again in such a situation.

    .. WARNING::
         Using this object to compute gradients when ``num_derivatives`` := 0 results in UNDEFINED BEHAVIOR.

    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :points_to_sample[dim][num_to_sample]: points at which to compute GP-derived quantities (mean, variance, etc.)
      :num_to_sample: number of points being sampled concurrently
      :num_derivatives: configure this object to compute ``num_derivatives`` derivative terms wrt
        points_to_sample[:][0:num_derivatives]; 0 means no gradient computation will be performed.
  \endrst*/
  PointsToSampleState(const GaussianProcess& gaussian_process,
                      double const * restrict points_to_sample_in,
                      int num_to_sample_in, int num_derivatives_in) OL_NONNULL_POINTERS;

  PointsToSampleState(PointsToSampleState&& other);

  /*!\rst
    Configures this object with new ``points_to_sample``.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that GaussianProcess's mean, variance (and gradients thereof) functions can be called.

    .. WARNING::
         This object's state is INVALIDATED if the gaussian_process used in SetupState is mutated!
         SetupState() should be called again in such a situation.

    .. WARNING::
         Using this object to compute gradients when ``num_derivatives`` := 0 results in UNDEFINED BEHAVIOR.

    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :points_to_sample[dim][num_to_sample]: points at which to compute GP-derived quantities (mean, variance, etc.)
      :num_to_sample: number of points being sampled concurrently
      :num_derivatives: configure this object to compute ``num_derivatives`` derivative terms wrt
        points_to_sample[:][0:num_derivatives]; 0 means no gradient computation will be performed.
  \endrst*/
  void SetupState(const GaussianProcess& gaussian_process, double const * restrict points_to_sample_in,
                  int num_to_sample_in, int num_derivatives_in) OL_NONNULL_POINTERS;

  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! number of points alerady sampled
  int num_sampled;
  //! number of points currently being sampled
  int num_to_sample;

  //! this object can compute ``num_derivatives`` derivative terms wrt
  //! points_to_sample[:][0:num_derivatives]; 0 means no gradient computation will be performed
  int num_derivatives;

  // state variables for predictive component
  //! points to make predictions about, ``Xs``
  std::vector<double> points_to_sample;

  // derived variables for predictive component; these are all *temporary* quantities
  //! the "mixed" covariance matrix: ``Ks, Ks_{ij}, K(X,Xs) = covariance(X_i, Xs_j)``, covariance matrix between training and test inputs (``num_sampled x num_to_sample``)
  std::vector<double> K_star;
  //! the gradient of mixed covariance matrix, ``Ks``, wrt ``Xs``
  std::vector<double> grad_K_star;
  //! the variance matrix (output from the GP)
  std::vector<double> V;
  //! ``K^{-1} Ks`` (computed without taking an inverse)
  std::vector<double> K_inv_times_K_star;
  //! the gradient of covariance(x_1, x_2) wrt x_1
  std::vector<double> grad_cov;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PointsToSampleState);
};

struct ExpectedImprovementState;
struct OnePotentialSampleExpectedImprovementState;

/*!\rst
  A class to encapsulate the computation of expected improvement and its spatial gradient. This class handles the
  general EI computation case using monte carlo integration; it can support q,p-EI optimization. It is designed to work
  with any GaussianProcess.  Additionally, this class has no state and within the context of EI optimization, it is
  meant to be accessed by const reference only.

  The random numbers needed for EI computation will be passed as parameters instead of contained as members to make
  multithreading more straightforward.
\endrst*/
class ExpectedImprovementEvaluator final {
 public:
  using StateType = ExpectedImprovementState;
  /*!\rst
    Constructs a ExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :num_mc_iterations: number of monte carlo iterations
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  ExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, int num_mc_iterations, double best_so_far);

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
    Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
    are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
    already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
    is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
    whereas ``f(Xs)`` are *GP-predicted* function evaluations.

    ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

    In words, we are computing the expected improvement (over the current ``best_so_far``, best known
    objective function value) that would result from sampling (aka running new experiments) at
    ``points_to_sample`` with ``points_being_sampled`` concurrent/ongoing experiments.

    In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.
    When faster (e.g., analytic) techniques are available, we will prefer them.

    The idea of the MC approach is to repeatedly sample at the union of ``points_to_sample`` and
    ``points_being_sampled``. This is analogous to gaussian_process_interface.sample_point_from_gp,
    but we sample ``num_union`` points at once:

    ``y = \mu + Lw``

    where ``\mu`` is the GP-mean, ``L`` is the ``chol_factor(GP-variance)`` and ``w`` is a vector
    of ``num_union`` draws from N(0, 1). Then:

    ``improvement_per_step = max(max(best_so_far - y), 0.0)``

    Observe that the inner ``max`` means only the smallest component of ``y`` contributes in each iteration.
    We compute the improvement over many random draws and average.

    .. Note:: These comments were copied into ExpectedImprovementInterface.compute_expected_improvement() in interfaces/expected_improvement_interface.py.

    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``normal_rng`` modified
    \return
      the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    Computes the (partial) derivatives of the expected improvement with respect to each point of ``points_to_sample``.
    As with ComputeExpectedImprovement(), this computation accounts for the effect of ``points_being_sampled``
    concurrent experiments.

    ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI..

    In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
    Monte-Carlo simulation to approximate it. When faster (e.g., analytic) techniques are available, we will prefer them.

    The MC computation of grad EI is similar to the computation of EI (decsribed in
    compute_expected_improvement). We differentiate ``y = \mu + Lw`` wrt ``points_to_sample``;
    only terms from the gradient of ``\mu`` and ``L`` contribute. In EI, we computed:

    ``improvement_per_step = max(max(best_so_far - y), 0.0)``

    and noted that only the smallest component of ``y`` may contribute (if it is > 0.0).
    Call this index ``winner``. Thus in computing grad EI, we only add gradient terms
    that are attributable to the ``winner``-th component of ``y``.

    .. Note:: These comments were copied into ExpectedImprovementInterface.compute_expected_improvement() in interfaces/expected_improvement_interface.py.

    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified; ``normal_rng`` modified
      :grad_EI[dim][num_to_sample]: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{d,i}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(ExpectedImprovementEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! number of monte carlo iterations
  int num_mc_iterations_;
  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;
  //! pointer to gaussian process used in EI computations
  const GaussianProcess * gaussian_process_;
};

/*!\rst
  State object for ExpectedImprovementEvaluator.  This tracks the points being sampled in concurrent experiments
  (``points_being_sampled``) ALONG with the points currently being evaluated via expected improvement for future experiments
  (called ``points_to_sample``); these are the p and q of q,p-EI, respectively.  ``points_to_sample`` joined with
  ``points_being_sampled`` is stored in ``union_of_points`` in that order.

  This struct also tracks the state of the GaussianProcess that underlies the expected improvement computation: the GP state
  is built to handle the initial ``union_of_points``, and subsequent updates to ``points_to_sample`` in this object also update
  the GP state.

  This struct also holds a pointer to a random number generator needed for Monte Carlo integrated EI computations.

  .. WARNING::
       Users MUST guarantee that multiple state objects DO NOT point to the same RNG (in a multithreaded env).

  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
struct ExpectedImprovementState final {
  using EvaluatorType = ExpectedImprovementEvaluator;

  /*!\rst
    Constructs an ExpectedImprovementState object with a specified source of randomness for the purpose of computing EI
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
      :normal_rng[1]: pointer to a properly initialized\* NormalRNG object

    .. NOTE::
         \* The NormalRNG object must already be seeded.  If multithreaded computation is used for EI, then every state object
         must have a different NormalRNG (different seeds, not just different objects).
  \endrst*/
  ExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                           double const * restrict points_being_sampled, int num_to_sample_in,
                           int num_being_sampled_in, bool configure_for_gradients, NormalRNGInterface * normal_rng_in);

  ExpectedImprovementState(ExpectedImprovementState&& other);

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
  static std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample,
                                                double const * restrict points_being_sampled,
                                                int num_to_sample, int num_being_sampled,
                                                int dim) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> union_of_points(dim*(num_to_sample + num_being_sampled));
    std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.data());
    std::copy(points_being_sampled, points_being_sampled + dim*num_being_sampled,
              union_of_points.data() + dim*num_to_sample);
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
  void SetCurrentPoint(const EvaluatorType& ei_evaluator,
                          double const * restrict points_to_sample) OL_NONNULL_POINTERS;

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
  void SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample);

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
  NormalRNGInterface * normal_rng;

  // temporary storage: preallocated space used by ExpectedImprovementEvaluator's member functions
  //! the mean of the GP evaluated at union_of_points
  std::vector<double> to_sample_mean;
  //! the gradient of the GP mean evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_mu;
  //! the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points
  std::vector<double> cholesky_to_sample_var;
  //! the gradient of the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_chol_decomp;

  //! improvement (per mc iteration) evaluated at each of union_of_points
  std::vector<double> EI_this_step_from_var;
  //! tracks the aggregate grad EI from all mc iterations
  std::vector<double> aggregate;
  //! normal rng draws
  std::vector<double> normals;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(ExpectedImprovementState);
};

/*!\rst
  This is a specialization of the ExpectedImprovementEvaluator class for when the number of potential samples is 1; i.e.,
  ``num_to_sample == 1`` and the number of concurrent samples is 0; i.e. ``num_being_sampled == 0``.
  In other words, this class only supports the computation of 1,0-EI.  In this case, we have analytic formulas
  for computing EI and its gradient.

  Thus this class does not perform any explicit numerical integration, nor do its EI functions require access to a
  random number generator.

  This class's methods have some parameters that are unused or redundant.  This is so that the interface matches that of
  the more general ExpectedImprovementEvaluator.

  For other details, see ExpectedImprovementEvaluator for more complete description of what EI is and the outputs of
  EI and grad EI computations.
\endrst*/
class OnePotentialSampleExpectedImprovementEvaluator final {
 public:
  using StateType = OnePotentialSampleExpectedImprovementState;

  //! Minimum allowed variance value in the "1D" analytic EI computation.
  //! Values that are too small result in problems b/c we may compute ``std_dev/var`` (which is enormous
  //! if ``std_dev = 1.0e-150`` and ``var = 1.0e-300``) since this only arises when we fail to compute ``std_dev = var = 0.0``.
  //! Note: this is only relevant if noise = 0.0; this minimum will not affect EI computation with noise since this value
  //! is below the smallest amount of noise users can meaningfully add.
  //! This is the smallest possible value that prevents the denominator (best_so_far - mean) / sqrt(variance)
  //! from being 0. 1D analytic EI is simple and no other robustness considerations are needed.
  static constexpr double kMinimumVarianceEI = std::numeric_limits<double>::min();

  //! Minimum allowed variance value in the "1D" analytic grad EI computation.
  //! See kMinimumVarianceEI for more details.
  //! This value was chosen so its sqrt would be a little larger than GaussianProcess::kMinimumStdDev (by ~12x).
  //! The 150.0 was determined by numerical experiment with the setup in EIOnePotentialSampleEdgeCasesTest
  //! in order to find a setting that would be robust (no 0/0) while introducing minimal error.
  static constexpr double kMinimumVarianceGradEI = 150.0*Square(GaussianProcess::kMinimumStdDev);

  /*!\rst
    Constructs a OnePotentialSampleExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  OnePotentialSampleExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, double best_so_far);

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
    Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``

    Uses analytic formulas to evaluate the expected improvement.

    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
    \return
      the expected improvement from sampling ``point_to_sample``
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const;

  /*!\rst
    Computes the (partial) derivatives of the expected improvement with respect to the point to sample.

    Uses analytic formulas to evaluate the spatial gradient of the expected improvement.

    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
      :grad_EI[dim]: gradient of EI, ``\pderiv{EI(x)}{x_d}``, where ``x`` is ``points_to_sample``
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(OnePotentialSampleExpectedImprovementEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim_;
  //! best (minimum) objective function value (in ``points_sampled_value``)
  double best_so_far_;

  //! normal distribution object
  const boost::math::normal_distribution<double> normal_;
  //! pointer to gaussian process used in EI computations
  const GaussianProcess * gaussian_process_;
};

/*!\rst
  State object for OnePotentialSampleExpectedImprovementEvaluator.  This tracks the *ONE* ``point_to_sample``
  being evaluated via expected improvement.

  This is just a special case of ExpectedImprovementState; see those class docs for more details.
  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
struct OnePotentialSampleExpectedImprovementState final {
  using EvaluatorType = OnePotentialSampleExpectedImprovementEvaluator;

  /*!\rst
    Constructs an OnePotentialSampleExpectedImprovementState object for the purpose of computing EI
    (and its gradient) over the specified point to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ``ei_evaluator``).

    .. WARNING::
         This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.

    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in UNDEFINED BEHAVIOR.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: point at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test point for GP predictions)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
  \endrst*/
  OnePotentialSampleExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                             double const * restrict point_to_sample_in, bool configure_for_gradients);

  /*!\rst
    Constructor wrapper to match the signature of the ctor for ExpectedImprovementState().
  \endrst*/
  OnePotentialSampleExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                             double const * restrict points_to_sample,
                                             double const * restrict OL_UNUSED(points_being_sampled),
                                             int OL_UNUSED(num_to_sample_in), int OL_UNUSED(num_being_sampled_in),
                                             bool configure_for_gradients, NormalRNGInterface * OL_UNUSED(normal_rng_in));

  OnePotentialSampleExpectedImprovementState(OnePotentialSampleExpectedImprovementState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  /*!\rst
    Get ``point_to_sample``: the potential future sample whose EI (and/or gradients) is being evaluated

    \output
      :point_to_sample[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict point_to_sample_out) const noexcept OL_NONNULL_POINTERS {
    std::copy(point_to_sample.begin(), point_to_sample.end(), point_to_sample_out);
  }

  /*!\rst
    Change the potential sample whose EI (and/or gradient) is being evaluated.
    Update the state's derived quantities to be consistent with the new point.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& ei_evaluator,
                          double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with a new ``point_to_sample``, the location of the potential sample whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.

    .. WARNING::
         This object's state is INVALIDATED if the ei_evaluator (including the GaussianProcess it depends on) used in
         SetupState is mutated! SetupState() should be called again in such a situation.

    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetupState(const EvaluatorType& ei_evaluator,
                  double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! number of points to sample (i.e., the "q" in q,p-EI); MUST be 1
  const int num_to_sample = 1;
  //! number of derivative terms desired (usually 0 for no derivatives or num_to_sample)
  const int num_derivatives;

  //! point at which to evaluate EI and/or its gradient (e.g., to check its value in future experiments)
  std::vector<double> point_to_sample;

  //! gaussian process state
  GaussianProcess::StateType points_to_sample_state;

  // temporary storage: preallocated space used by OnePotentialSampleExpectedImprovementEvaluator's member functions
  //! the gradient of the GP mean evaluated at point_to_sample, wrt point_to_sample
  std::vector<double> grad_mu;
  //! the gradient of the sqrt of the GP variance evaluated at point_to_sample wrt point_to_sample
  std::vector<double> grad_chol_decomp;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(OnePotentialSampleExpectedImprovementState);
};

/*!\rst
  Set up vector of OnePotentialSampleExpectedImprovementEvaluator::StateType.

  This is a utility function just for reducing code duplication.

  dim is the spatial dimension, ``ei_evaluator.dim()``

  \param
    :ei_evaluator: evaluator object associated w/the state objects being constructed
    :starting_point[dim]: initial point to load into state (must be a valid point for the problem)
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
  \output
    :state_vector[max_num_threads]: vector of states containing ``max_num_threads`` properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupExpectedImprovementState(
    const OnePotentialSampleExpectedImprovementEvaluator& ei_evaluator,
    double const * restrict starting_point,
    int max_num_threads,
    bool configure_for_gradients,
    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(ei_evaluator, starting_point, configure_for_gradients);
  }
}

/*!\rst
  Set up vector of ExpectedImprovementEvaluator::StateType.

  This is a utility function just for reducing code duplication.

  \param
    :ei_evaluator: evaluator object associated w/the state objects being constructed
    :points_to_sample[dim][num_to_sample]: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrently experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
    :normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration
  \output
    :state_vector[max_num_threads]: vector of states containing ``max_num_threads`` properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupExpectedImprovementState(
    const ExpectedImprovementEvaluator& ei_evaluator,
    double const * restrict points_to_sample,
    double const * restrict points_being_sampled,
    int num_to_sample,
    int num_being_sampled,
    int max_num_threads,
    bool configure_for_gradients,
    NormalRNG * normal_rng,
    std::vector<typename ExpectedImprovementEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(ei_evaluator, points_to_sample, points_being_sampled, num_to_sample,
                               num_being_sampled, configure_for_gradients, normal_rng + i);
  }
}

/*!\rst
  Solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or header docs) by optimizing the Expected Improvement.
  Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``.  Please see that file for details on gradient descent and see gpp_optimizer_parameters.hpp
  for the meanings of the GradientDescentParameters.

  This function is just a simple wrapper that sets up the Evaluator's State and calls a general template for restarted GD.

  This function does not perform multistarting or employ any other robustness-boosting heuristcs; it only
  converges if the ``initial_guess`` is close to the solution. In general,
  ComputeOptimalPointsToSample() (see below) is preferred. This function is meant for:

  1. easier testing;
  2. if you really know what you're doing.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  \param
    :ei_evaluator: reference to object that can compute ExpectedImprovement and its spatial gradient
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :initial_guess[dim][num_to_sample]: initial guess for gradient descent
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :normal_rng[1]: a NormalRNG object that provides the (pesudo)random source for MC integration
  \output
    :normal_rng[1]: NormalRNG object will have its state changed due to random draws
    :next_point[dim][num_to_sample]: points yielding the best EI according to gradient descent
\endrst*/
template <typename ExpectedImprovementEvaluator, typename DomainType>
void RestartedGradientDescentEIOptimization(const ExpectedImprovementEvaluator& ei_evaluator,
                                            const GradientDescentParameters& optimizer_parameters,
                                            const DomainType& domain, double const * restrict initial_guess,
                                            double const * restrict points_being_sampled, int num_to_sample,
                                            int num_being_sampled, NormalRNG * normal_rng,
                                            double * restrict next_point) {
  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
    return;
  }
  int dim = ei_evaluator.dim();

  OL_VERBOSE_PRINTF("Expected Improvement Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  bool configure_for_gradients = true;
  typename ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, initial_guess,
                                                            points_being_sampled, num_to_sample,
                                                            num_being_sampled, configure_for_gradients,
                                                            normal_rng);

  using RepeatedDomain = RepeatedDomain<DomainType>;
  RepeatedDomain repeated_domain(domain, num_to_sample);
  GradientDescentOptimizer<ExpectedImprovementEvaluator, RepeatedDomain> gd_opt;
  gd_opt.Optimize(ei_evaluator, optimizer_parameters, repeated_domain, &ei_state);
  ei_state.GetCurrentPoint(next_point);
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs).  Starts a GD run from each point in ``start_point_set``.  The point corresponding to the
  optimal EI\* is stored in ``best_next_point``.

  \* Multistarting is heuristic for global optimization. EI is not convex so this method may not find the true optimum.

  This function wraps MultistartOptimizer<>::MultistartOptimize() (see ``gpp_optimization.hpp``), which provides the multistarting
  component. Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``. Please see that file for details on gradient descent and see ``gpp_optimizer_parameters.hpp``
  for the meanings of the GradientDescentParameters.

  This function (or its wrappers, e.g., ComputeOptimalPointsToSampleWithRandomStarts) are the primary entry-points for
  gradient descent based EI optimization in the ``optimal_learning`` library.

  Users may prefer to call ComputeOptimalPointsToSample(), which applies other heuristics to improve robustness.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING::
       This function fails ungracefully if NO improvement can be found!  In that case,
       ``best_next_point`` will always be the first point in ``start_point_set``.
       ``found_flag`` will indicate whether this occured.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :start_point_set[dim][num_to_sample][num_multistarts]: set of initial guesses for MGD (one block of num_to_sample points per multistart)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points in set of initial guesses
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero EI
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void ComputeOptimalPointsToSampleViaMultistartGradientDescent(
    const GaussianProcess& gaussian_process,
    const GradientDescentParameters& optimizer_parameters,
    const DomainType& domain,
    const ThreadSchedule& thread_schedule,
    double const * restrict start_point_set,
    double const * restrict points_being_sampled,
    int num_multistarts,
    int num_to_sample,
    int num_being_sampled,
    double best_so_far,
    int max_int_steps,
    NormalRNG * normal_rng,
    bool * restrict found_flag,
    double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  bool configure_for_gradients = true;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, thread_schedule.max_num_threads,
                                  configure_for_gradients, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, start_point_set);

    GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimizer_parameters,
                                            domain, thread_schedule, start_point_set,
                                            num_multistarts,
                                            ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);

    std::vector<typename ExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, points_being_sampled,
                                  num_to_sample, num_being_sampled, thread_schedule.max_num_threads,
                                  configure_for_gradients, normal_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, start_point_set);

    using RepeatedDomain = RepeatedDomain<DomainType>;
    RepeatedDomain repeated_domain(domain, num_to_sample);
    GradientDescentOptimizer<ExpectedImprovementEvaluator, RepeatedDomain> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<ExpectedImprovementEvaluator, RepeatedDomain> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimizer_parameters,
                                            repeated_domain, thread_schedule, start_point_set,
                                            num_multistarts,
                                            ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs), starting from ``num_multistarts`` points selected randomly from the within th domain.

  This function is a simple wrapper around ComputeOptimalPointsToSampleViaMultistartGradientDescent(). It additionally
  generates a set of random starting points and is just here for convenience when better initial guesses are not
  available.

  See ComputeOptimalPointsToSampleViaMultistartGradientDescent() for more details.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalPointsToSampleWithRandomStarts(const GaussianProcess& gaussian_process,
                                                  const GradientDescentParameters& optimizer_parameters,
                                                  const DomainType& domain, const ThreadSchedule& thread_schedule,
                                                  double const * restrict points_being_sampled,
                                                  int num_to_sample, int num_being_sampled, double best_so_far,
                                                  int max_int_steps, bool * restrict found_flag,
                                                  UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng,
                                                  double * restrict best_next_point) {
  std::vector<double> starting_points(gaussian_process.dim()*optimizer_parameters.num_multistarts*num_to_sample);

  // GenerateUniformPointsInDomain() is allowed to return fewer than the requested number of multistarts
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  int num_multistarts = repeated_domain.GenerateUniformPointsInDomain(optimizer_parameters.num_multistarts,
                                                                      uniform_generator, starting_points.data());

  ComputeOptimalPointsToSampleViaMultistartGradientDescent(gaussian_process, optimizer_parameters, domain,
                                                           thread_schedule, starting_points.data(),
                                                           points_being_sampled, num_multistarts, num_to_sample,
                                                           num_being_sampled, best_so_far, max_int_steps,
                                                           normal_rng, found_flag, best_next_point);
#ifdef OL_WARNING_PRINT
  if (false == *found_flag) {
    OL_WARNING_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
    OL_WARNING_PRINTF("First multistart point was returned:\n");
    PrintMatrixTrans(starting_points.data(), num_to_sample, gaussian_process.dim());
  }
#endif
}

/*!\rst
  Function to evaluate Expected Improvement (q,p-EI) over a specified list of ``num_multistarts`` points.
  Optionally outputs the EI at each of these points.
  Outputs the point of the set obtaining the maximum EI value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
  This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

  This function is just a wrapper that builds the required state objects and a NullOptimizer object and calls
  MultistartOptimizer<...>::MultistartOptimize(...); see gpp_optimization.hpp.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :initial_guesses[dim][num_to_sample][num_multistarts]: list of points at which to compute EI
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :function_values[num_multistarts]: EI evaluated at each point of ``initial_guesses``, in the same order as
      ``initial_guesses``; never dereferenced if nullptr
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
void EvaluateEIAtPointList(const GaussianProcess& gaussian_process,
                           const ThreadSchedule& thread_schedule,
                           double const * restrict initial_guesses,
                           double const * restrict points_being_sampled,
                           int num_multistarts, int num_to_sample,
                           int num_being_sampled, double best_so_far,
                           int max_int_steps,
                           bool * restrict found_flag, NormalRNG * normal_rng,
                           double * restrict function_values,
                           double * restrict best_next_point);

/*!\rst
  Perform a random, naive search to "solve" the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs).  Evaluates EI at ``num_multistarts`` points (e.g., on a latin hypercube) to find the
  point with the best EI value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  Wraps EvaluateEIAtPointList(); constructs the input point list with a uniform random sampling from the given Domain object.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of random points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to dumb search
\endrst*/
template <typename DomainType>
void ComputeOptimalPointsToSampleViaLatinHypercubeSearch(const GaussianProcess& gaussian_process,
                                                         const DomainType& domain,
                                                         const ThreadSchedule& thread_schedule,
                                                         double const * restrict points_being_sampled,
                                                         int num_multistarts, int num_to_sample,
                                                         int num_being_sampled, double best_so_far,
                                                         int max_int_steps,
                                                         bool * restrict found_flag,
                                                         UniformRandomGenerator * uniform_generator,
                                                         NormalRNG * normal_rng,
                                                         double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts*num_to_sample);
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  num_multistarts = repeated_domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator,
                                                                  initial_guesses.data());

  EvaluateEIAtPointList(gaussian_process, thread_schedule, initial_guesses.data(),
                        points_being_sampled, num_multistarts, num_to_sample,
                        num_being_sampled, best_so_far, max_int_steps,
                        found_flag, normal_rng, nullptr, best_next_point);
}

/*!\rst
  Solve the q,p-EI problem (see header docs) by optimizing the Expected Improvement.
  Uses multistart gradient descent, "dumb" search, and/or other heuristics to perform the optimization.

  This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
  improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

  Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
  experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
  (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
  (requires monte-carlo iteration), so this method is usually very expensive.

  Wraps ComputeOptimalPointsToSampleWithRandomStarts() and ComputeOptimalPointsToSampleViaLatinHypercubeSearch().

  Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
  makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
  GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

  .. NOTE:: These comments were copied into multistart_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
    :num_lhc_samples: number of samples to draw if/when doing latin hypercube search
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_points_to_sample corresponds to a nonzero EI if sampled simultaneously
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_points_to_sample[num_to_sample*dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalPointsToSample(const GaussianProcess& gaussian_process,
                                  const GradientDescentParameters& optimizer_parameters,
                                  const DomainType& domain, const ThreadSchedule& thread_schedule,
                                  double const * restrict points_being_sampled,
                                  int num_to_sample, int num_being_sampled, double best_so_far,
                                  int max_int_steps, bool lhc_search_only,
                                  int num_lhc_samples, bool * restrict found_flag,
                                  UniformRandomGenerator * uniform_generator,
                                  NormalRNG * normal_rng, double * restrict best_points_to_sample);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
extern template void ComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps,
    bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_MATH_HPP_
