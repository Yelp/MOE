// gpp_math.hpp
/**\rst
  These comments are quite long; here's a table of contents:

  1. OVERVIEW OF GAUSSIAN PROCESSES AND EXPECTED IMPROVEMENT; WHAT ARE WE TRYING TO DO?
  2. FILE OVERVIEW
  3. IMPLEMENTATION NOTES
  4. NOTATION
  5. CITATIONS

  **1 OVERVIEW OF GAUSSIAN PROCESSES AND EXPECTED IMPROVEMENT; WHAT ARE WE TRYING TO DO?**

  At a high level, this file optimizes an objective function \ms f(x)\me.  This operation
  requires data/uncertainties about prior and concurrent experiments as well as
  a covariance function describing how these data [are expected to] relate to each
  other.  The points \ms x\me represent experiments. If \ms f(x)\me is say, survival rate for
  a drug test, the dimensions of x might include dosage amount, dosage frequency,
  and overall drug-use time-span.

  The objective function is not required in closed form; instead, only the ability
  to sample it at points of interest is needed.  Thus, the optimization process
  cannot work with \ms f(x)\me directly; instead a surrogate is built via interpolation
  with Gaussian Proccesses (GPs).

  Following Rasmussen & Williams (Chapter 2, Section 2), a Gaussian Process is a collection of random
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
  algorithm believes is the most likely outcome from sampling a particular point
  (aka conducting a particular experiment).

  See ExpectedImprovementEvaluator and OnePotentialSampleExpectedImprovementEvaluator class
  docs for further details on computing EI.  Both support ComputeExpectedImprovement() and
  ComputeGradExpectedImprovement().

  The dimension of the GP is equal to the number of simultaneous experiments being run;
  i.e., the GP may be multivariate.  The behavior of the GP is controlled by its underlying
  covariance function and the data/uncertainty of prior points (experiments).

  With the ability the compute EI, the final step is to optimize
  to find the best EI.  This is done using multistart gradient descent (MGD), in
  ComputeOptimalPointToSampleWithRandomStarts().  This generates a uniform random
  sampling of points and calls ComputeOptimalPointToSampleViaMultistartGradientDescent(),
  which carries out the multistart process via templates from gpp_optimization.hpp.

  The idea behind gradient descent is simple.  The gradient gives us the direction of
  steepest ascent (negative gradient is steepest descent).  So each iteration, we compute
  the gradient and take a step in that direction.  The size of the step is not specified
  by GD and is left to the specific implementation.  Basically if we take steps that are
  too large, we run the risk of over-shooting the solution and even diverging.  If we
  take steps that are too small, it may take an intractably long time to reach the solution.
  Thus the magic is in choosing the step size; we do not claim that our implementation is
  perfect, but it seems to work reasonably.  See gpp_optimization.hpp for more details about
  GD as well as the template definition.

  For particularly difficult problems or problems where gradient descent's parameters are not
  well-chosen, GD can fail to converge.  If this happens, we can fall back to a 'dumb' search
  (i.e., evaluate EI at a large number of random points and take the best one).  This
  functionality is accessed through: ComputeOptimalPointToSampleViaLatinHypercubeSearch<>().

  **2 FILE OVERVIEW**

  This file contains mathematical functions supporting optimal learning.
  These include functions to compute characteristics of Gaussian Processes
  (e.g., variance, mean) and the gradients of these quantities as well as functions to
  compute and optimize the expected improvement.

  Functions here generally require some combination of a CovarianceInterface object as well as
  data about prior and current (i.e., concurrent) experiments.  These data are encapsulated in
  the GaussianProcess class.  Then we build an ExpectedImprovementEvaluator object (with
  associated state, see gpp_common.hpp item 5 for (Evaluator, State) relations) on top of a
  GaussianProcess for computing and optimizing EI.

  For further theoretical details about Gaussian Processes, see
  Rasmussen and Williams, Gaussian Processes for Machine Learning (2006).
  A bare-bones summary is provided in gpp_math.cpp.

  For further details about expected improvement and the optimization thereof,
  see Scott Clark's PhD thesis.  Again, a summary is provided in gpp_math.cpp's file comments.

  **3 IMPLEMENTATION NOTES**

  a. This file has a few primary endpoints for EI optimization:

     i. ComputeOptimalPointToSampleWithRandomStarts<>()
          Takes in a covariance function and prior data, outputs the next best point (experiment)
          to sample (run). Uses gradient descent. Only produces a single new point to sample.
     ii. ComputeOptimalPointToSampleViaLatinHypercubeSearch<>()
          Takes in a covariance function and prior data, outputs the next best point (experiment)
          to sample (run). Uses 'dumb' search. Only produces a single new point to sample.
     iii. ComputeOptimalSetOfPointsToSample<>()
          Takes in a covariance function and prior data, outputs the next best set of points (experiments)
          to sample (run). Uses gradient descent and/or 'dumb' search. Produces a specified number of
          points for simultaneous sampling.

     .. NOTE::
         See gpp_math.cpp's header comments for more detailed implementation notes.

         There are also several other functions with external linkage in this header; these
         are provided to ease testing and to permit lower level access from python.

  b. See gpp_common.hpp header comments for additional implementation notes.

  **4 NOTATION**

  And domain-specific notation, following Rasmussen, Williams:
      * X = points_sampled; this is the training data (size dim X num_sampled), also called the design matrix
      * Xs = points_to_sample; this is the test data (size dim X num_to_sample)
      * y, f, \ms f(x)\me = points_sampled_value, the experimental results from sampling training points
      * K, K(X,X) = covariance(X_i, X_j), covariance matrix between training inputs
      * Ks, K(Xs,X) = covariance(X_i, Xs_j), covariance matrix between training and test inputs
      * Kss, K(Xs,Xs) = covariance(Xs_i, Xs_j), covariance matrix between test inputs
      * \theta: (vector) of hyperparameters for a covariance function

  .. NOTE::
      Due to confusion with multiplication, Rasmussen & Williams' "K_*" notation has been repalced with "Ks" and
      "K_{**}" is "Kss".

  **5 CITATIONS**

  a)
  Gaussian Processes for Machine Learning
  Carl edward Rasmussen and Christopher K. I. Williams. 2006.
  Massachusetts Institute of Technology.  55 Hayward St., Cambridge, MA 02142.
  http://www.gaussianprocess.org/gpml/ (free electronic copy)

  b)
  Parallel Machine Learning Algorithms In Bioinformatics and Global Optimization (PhD Dissertation)
  Part II, EPI: Expected Parallel Improvement
  Scott Clark. 2012.
  Cornell University, Center for Applied Mathematics.  Ithaca, NY.
  https://github.com/sc932/Thesis
  sclark@yelp.com

  c)
  Differentiation of the Cholesky Algorithm
  S. P. Smith. 1995.
  Journal of Computational and Graphical Statistics. Volume 4. Number 2. p134-147

  d)
  A Multi-points Criterion for Deterministic Parallel Global Optimization based on Gaussian Processes.
  David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.  2008.
  D´epartement 3MI. Ecole Nationale Sup´erieure des Mines. 158 cours Fauriel, Saint-Etienne, France.
  {ginsbourger, leriche, carraro}@emse.fr

  e)
  Efficient Global Optimization of Expensive Black-Box Functions
  Jones, D.R., Schonlau, M., Welch, W.J. 1998.
  Journal of Global Optimization, 13, 455-492.
 \endrst*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MATH_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MATH_HPP_

namespace optimal_learning_gpp_math_hpp {
}

#include <algorithm>
#include <memory>
#include <vector>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_covariance.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

// forward declaration to use PointsToSampleState pointers/references in GaussianProcess
struct PointsToSampleState;

/**\rst
  Object that encapsulates Gaussian Process Priors (GPPs).  A GPP is defined by a set of
  (sample point, function value, noise variance) triples along with a covariance function that relates the points.
  Each point has dimension dim.  These are the training data; for example, each sample point might specify an experimental
  cohort and the corresponding function value is the objective measured for that experiment.  There is one noise variance
  value per function value; this is the measurement error and is treated as N(0, noise_variance) Gaussian noise.

  GPPs estimate a real process \ms f(x)\me = GP(m(x), k(x,x')) (see file docs).  This class deals with building an estimator
  to the actual process using measurements taken from the actual process--the (sample point, function val, noise) triple.
  Then predictions about unknown points can be made by sampling from the GPP--in particular, finding the (predicted)
  mean and variance.  These functions (and their gradients) are provided in ComputeMeanOfPoints, ComputeVarianceOfPoints,
  etc.

  Further mathematical details are given in the implementation comments, but we are essentially computing:
  ComputeMeanOfPoints    : K(Xs, X) * [K(X,X) + \sigma_n^2 I]^{-1} * y
  ComputeVarianceOfPoints: K(Xs, Xs) - K(Xs,X) * [K(X,X) + \sigma_n^2 I]^{-1} * K(X,Xs)
  This (estimated) mean and variance characterize the predicted distributions of the actual m(x), k(x,x') functions
  that underly our GP.

  For testing and experimental purposes, this class provides a framework for sampling points from the GP (i.e., given a
  point to sample and predicted measurement noise) as well as adding additional points to an already-formed GP.  Sampling
  points requires drawing from N(0,1) so this class also holds PRNG state to do so via the NormalRNG object from gpp_random.

  .. NOTE::
    functions that manipulate the PRNG directly or indirectly (changing state, generating points)
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

  static constexpr EngineType::result_type kDefaultSeed = 87214;

  /**\rst
    Constructs a GaussianProcess object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \params
        :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
        :points_sampled[dim][num_sampled]: points that have already been sampled
        :points_sampled_value[num_sampled]: values of the already-sampled points
        :noise_variance[num_sampled]: the \sigma_n^2 (noise variance) associated w/observation, points_sampled_value
        :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
        :num_sampled: number of already-sampled points
  \endrst*/
  GaussianProcess(const CovarianceInterface& covariance_in, double const * restrict points_sampled_in, double const * restrict points_sampled_value_in, double const * restrict noise_variance_in, int dim_in, int num_sampled_in) OL_NONNULL_POINTERS :
      dim_(dim_in),
      num_sampled_(num_sampled_in),
      covariance_ptr_(covariance_in.Clone()),
      covariance_(*covariance_ptr_),
      points_sampled_(points_sampled_in, points_sampled_in + num_sampled_in*dim_in),
      points_sampled_value_(points_sampled_value_in, points_sampled_value_in + num_sampled_in),
      noise_variance_(noise_variance_in, noise_variance_in + num_sampled_),
      K_chol_(num_sampled_in*num_sampled_in),
      K_inv_y_(num_sampled_),
      normal_rng_(kDefaultSeed) {
    RecomputeDerivedVariables();
  }

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_sampled() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_sampled_;
  }

  /**\rst
    Change the hyperparameters of this GP's covariance function.
    Also forces recomputation of all derived quantities for GP to remain consistent.

    \params
        :hyperparameters_new[covariance_.GetNumberOfHyperparameters]: new hyperparameter array
  \endrst*/
  void SetCovarianceHyperparameters(double const * restrict hyperparameters_new) OL_NONNULL_POINTERS {
    covariance_.SetHyperparameters(hyperparameters_new);
    RecomputeDerivedVariables();
  }

  /**\rst
    Sets up the PointsToSampleState object so that it can be used to compute GP mean, variance, and gradients thereof.
    ASSUMES all needed space is ALREADY ALLOCATED.

    This function should not be called directly; instead use PointsToSampleState::SetupState().

    \params
        :points_to_sample_state[1]: pointer to a PointsToSampleState object where all space has been properly allocated
        :configure_for_gradients: whether the resulting PointsToSampleState object should be configured for gradient computation
    
    \return
        points_to_sample_state[1]: pointer to a fully configured PointsToSampleState object. overwrites input
  \endrst*/
  void FillPointsToSampleState(StateType * points_to_sample_state, bool configure_for_gradients) const OL_NONNULL_POINTERS;

  /**\rst
    Adds a single (point, fcn value) pair to the GP with the option of noise variance (set to 0.0 if undesired).

    Also forces recomputation of all derived quantities for GP to remain consistent.

    \params
        :new_point[dim]: coordinates of the new point to add
        :new_point_value: function value at the new point
        :noise_variance: \sigma_n^2 corresponding to the signal noise in measuring new_point_value
  \endrst*/
  void AddPointToGP(double const * restrict new_point, double new_point_value, double noise_variance) OL_NONNULL_POINTERS;

  /**\rst
    Sample a function value from a Gaussian Process prior, provided a point at which to sample.

    Uses the formula function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2, where w1, w2 are draws from N(0,1).

    Note: set noise_variance to 0 if you want "accurate" draws from the GP.
    BUT if the drawn (point, value) pair is meant to be added back into the GP (e.g., for testing), then this point
    MUST be drawn with noise_variance equal to the noise associated with "point" as a member of "points_sampled"

    \params
        :point_to_sample[dim]: coordinates of the point at which to generate a function value (from GP)
        :noise_variance_this_point: if this point is to be added into the GP, it needs to be generated with its associated noise var
    \return
        function value drawn from this GP
  \endrst*/
  double SamplePointFromGP(double const * restrict point_to_sample, double noise_variance_this_point) noexcept OL_NONNULL_POINTERS;

  /**\rst
    Computes the mean of a GP.
    points_to_sample and points_sampled are not allowed to contain duplicates either within
    themselves or with each other.  Violating this results in undefined behavior.
    No inputs may be nullptr.

    \param
        :points_to_sample_state: a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
    
    \return
        :mean_of_points[num_to_sample]: mean of GP, one per GP dimension
  \endrst*/
  void ComputeMeanOfPoints(const StateType& points_to_sample_state, double * restrict mean_of_points) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Computes the gradient of the mean of a GP with respect to Xs, points_to_sample.
    points_to_sample and points_sampled are not allowed to contain duplicates either within
    themselves or with each other.  Violating this results in undefined behavior.
    No inputs may be nullptr.

    Note that grad_mu is nominally sized: `grad_mu[dim][num_to_sample][num_to_sample]`.
    However, for `0 <= i, j < num_to_sample, i != j, grad_mu[d][i][j] = 0`.
    (See sources or implementation for further details.)
    Thus, grad_mu is stored in a reduced form which only tracks the nonzero entries.

    \params
        :points_to_sample_state: a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)

    \return
        :grad_mu[dim][num_to_sample]: gradient of the mean of the GP.  grad_mu[d][i] is
        actually the gradient of mu_i with respect to x_{d,i}, the d-th dimension of
        the i-th entry of points_to_sample.
  \endrst*/
  void ComputeGradMeanOfPoints(const StateType& points_to_sample_state, double * restrict grad_mu) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Computes the variance (matrix) of a GP.
    The variance matrix is symmetric and is stored in the LOWER TRIANGLE.
    points_to_sample and points_sampled are not allowed to contain duplicates either within
    themselves or with each other.  Violating this results in undefined behavior.
    No inputs may be nullptr.

    \params
        points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
        \return
        points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState; only temporary state may be mutated
        var_star[num_to_sample][num_to_sample]: variance of GP
  \endrst*/
  void ComputeVarianceOfPoints(StateType * points_to_sample_state, double * restrict var_star) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Similar to ComputeGradCholeskyVarianceOfPoints() except this does not include the gradient terms from
    the cholesky factorization.  Description will not be duplicated here.
  \endrst*/
  void ComputeGradVarianceOfPoints(StateType * points_to_sample_state, int var_of_grad, double * restrict grad_var) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Computes the gradient of the variance of a GP with respect to points_to_sample.
    This function also accounts for the effect on the gradient resulting from
    cholesky-factoring the variance matrix.  See Smith 1995 for algorithm details.

    points_to_sample and points_sampled are not allowed to contain duplicates either within
    themselves or with each other.  Violating this results in undefined behavior.
    No inputs may be nullptr.

    Note that grad_chol is nominally sized:
    grad_chol[dim][num_to_sample][num_to_sample][num_to_sample].
    Let this be indexed grad_chol[d][k][i][j], which is read the derivative of var[i][j]
    with respect to x_{d,k} (x = points_to_sample)

    Due to actual usage patterns, the full gradient tensor is never required simultaneously;
    thus only grad_chol[d][i][j] is formed with k (var_of_grad) as an input parameter to this function.

    \param
        points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState (configure via PointsToSampleState::SetupState)
        var_of_grad: index of points_to_sample in {0, .. num_to_sample-1} to be differentiated against

    \return
        points_to_sample_state[1]: ptr to a FULLY CONFIGURED PointsToSampleState; only temporary state may be mutated
        grad_chol[dim][num_to_sample][num_to_sample]: gradient of the cholesky-factored
        variance of the GP.  grad_chol[d][i][j] is actually the gradients of var_{i,j} with
        respect to x_{d,k}, the d-th dimension of the k-th entry of points_to_sample, where
        k = var_of_grad
  \endrst*/
  void ComputeGradCholeskyVarianceOfPoints(StateType * points_to_sample_state, int var_of_grad, double const * restrict chol_var, double * restrict grad_chol) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Re-seed the random number generator with the specified seed.
    See gpp_random, struct NormalRNG for details.

    \param
        seed: new seed to set
  \endrst*/
  void SetExplicitSeed(EngineType::result_type seed) noexcept {
    normal_rng_.SetExplicitSeed(seed);
  }

  /**\rst
    Re-seed the random number generator using a combination of the specified seed,
    current time, and potentially other factors.
    See gpp_random, struct NormalRNG for details.

    \param
        seed: base value for new seed
  \endrst*/
  void SetRandommizedSeed(EngineType::result_type seed) noexcept {
    normal_rng_.SetRandomizedSeed(seed, 0);  // this is intended for single-threaded use only, so thread_id = 0
  }

  /**\rst
    Reseeds the generator with its last used seed value.
    Useful for testing--e.g., can conduct multiple runs with the same initial conditions
  \endrst*/
  void ResetToMostRecentSeed() noexcept {
    normal_rng_.ResetToMostRecentSeed();
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(GaussianProcess);

 private:
  void BuildCovarianceMatrixWithNoiseVariance() noexcept;
  void BuildMixCovarianceMatrixTrans(double const * restrict points_to_sample, int num_to_sample, double * restrict cov_mat) const noexcept OL_NONNULL_POINTERS;

  /**\rst
    Recomputes (including resizing as needed) the derived quantities in this class.
    This function should be called any time state variables are changed.
  \endrst*/
  void RecomputeDerivedVariables() {
    // resize if needed
    if (unlikely(static_cast<int>(K_inv_y_.size()) != num_sampled_)) {
      K_chol_.resize(num_sampled_*num_sampled_);
      K_inv_y_.resize(num_sampled_);
    }

    // recompute derived quantities
    BuildCovarianceMatrixWithNoiseVariance();
    ComputeCholeskyFactorL(num_sampled_, K_chol_.data());

    std::copy(points_sampled_value_.begin(), points_sampled_value_.end(), K_inv_y_.begin());
    CholeskyFactorLMatrixVectorSolve(K_chol_.data(), num_sampled_, K_inv_y_.data());
  }


  // size information
  int dim_;  // spatial dimension (e.g., entries per point of points_sampled)
  int num_sampled_;  // number of points in points_sampled

  // state variables for prior
  std::unique_ptr<CovarianceInterface> covariance_ptr_;  // covariance class (for computing covariance and its gradients)
  CovarianceInterface& covariance_;  // reference to covariance object for convenience
  std::vector<double> points_sampled_;  // coordinates of already-sampled points, X
  std::vector<double> points_sampled_value_;  // function values at points_sampled, y
  std::vector<double> noise_variance_;  // \sigma_n^2, the noise variance

  // derived variables for prior
  std::vector<double> K_chol_;  // cholesky factorization of K (i.e., K(X,X) covariance matrix (prior), includes noise variance when relevant)
  std::vector<double> K_inv_y_;  // K^-1 * y; computed WITHOUT forming K^-1

  // Normal PRNG for use with sampling points from GP
  NormalGeneratorType normal_rng_;
};

/**\rst
  This object holds the state needed for a GaussianProcess object characterize the distribution of function values arising from
  sampling the GP at a list of points to sample.  This object is required by the GaussianProcess to access functionality for
  computing the mean, variance, and spatial gradients thereof.

  Once constructed, this object provides the SetupState() function to update it for computations at different sets of
  potential points to sample.

  See general comments on State structs in gpp_common.hpp's header docs.
\endrst*/
struct PointsToSampleState final {
  /**\rst
    Constructs a PointsToSampleState object with new points_to_sample.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that GaussianProcess's mean, variance (and gradients thereof) functions can be called.

    .. WARNING::
        This object's state is INVALIDATED if the gaussian_process used in construction is mutated!
        SetupState() should be called again in such a situation.

    .. WARNING::
        Using this object to compute gradients when configure_for_gradients := false results in UNDEFINED BEHAVIOR.

    \param
        gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
        points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
        num_to_sample: number of points being sampled concurrently
        configure_for_gradients: true if this object will be used to compute gradients, false otherwise
  \endrst*/
  PointsToSampleState(const GaussianProcess& gaussian_process, double const * restrict points_to_sample_in, int num_to_sample_in, bool configure_for_gradients_in) OL_NONNULL_POINTERS :
      dim(gaussian_process.dim()),
      num_sampled(gaussian_process.num_sampled()),
      num_to_sample(num_to_sample_in),
      configure_for_gradients(configure_for_gradients_in),
      points_to_sample(dim*num_to_sample),
      K_star_T(num_to_sample*num_sampled),
      grad_K_star(num_to_sample*num_sampled*dim),
      V(num_to_sample*num_sampled),
      K_inv_times_K_star_T(num_to_sample*num_sampled),
      grad_cov(dim),
      grad_mix_cov(num_sampled*dim) {
    SetupState(gaussian_process, points_to_sample_in, num_to_sample_in, configure_for_gradients_in);
  }

  PointsToSampleState(PointsToSampleState&& OL_UNUSED(other)) = default;

  /**\rst
    Configures this object with new points_to_sample.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all state variables so that GaussianProcess's mean, variance (and gradients thereof) functions can be called.

    .. WARNING::
        This object's state is INVALIDATED if the gaussian_process used in SetupState is mutated!
        SetupState() should be called again in such a situation.

    .. WARNING::
        Using this object to compute gradients when configure_for_gradients := false results in UNDEFINED BEHAVIOR.

    \param
        gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
        points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
        num_to_sample: number of points being sampled concurrently
        configure_for_gradients: true if this object will be used to compute gradients, false otherwise
  \endrst*/
  void SetupState(const GaussianProcess& gaussian_process, double const * restrict points_to_sample_in, int num_to_sample_in, bool configure_for_gradients_in) OL_NONNULL_POINTERS {
    configure_for_gradients = configure_for_gradients_in;
    // resize data depending on to sample points
    if (unlikely(num_to_sample != num_to_sample_in)) {
      num_to_sample = num_to_sample_in;  // update size
      // resize vectors
      points_to_sample.resize(dim*num_to_sample);
      K_star_T.resize(num_to_sample*num_sampled);
      grad_K_star.resize(num_to_sample*num_sampled*dim);
      V.resize(num_to_sample*num_sampled);
      K_inv_times_K_star_T.resize(num_to_sample*num_sampled);
    }

    // resize data depending on sampled points
    if (unlikely(num_sampled != gaussian_process.num_sampled())) {
      num_sampled = gaussian_process.num_sampled();
      K_star_T.resize(num_to_sample*num_sampled);
      grad_K_star.resize(num_to_sample*num_sampled*dim);
      V.resize(num_to_sample*num_sampled);
      K_inv_times_K_star_T.resize(num_to_sample*num_sampled);
      grad_mix_cov.resize(num_sampled*dim);
    }

    // set new points to sample
    std::copy(points_to_sample_in, points_to_sample_in + dim*num_to_sample, points_to_sample.begin());

    gaussian_process.FillPointsToSampleState(this, configure_for_gradients);
  }

  const int dim;  // spatial dimension (e.g., entries per point of points_sampled)
  int num_sampled;  // number of points alerady sampled
  int num_to_sample;  // number of points currently being sampled

  bool configure_for_gradients;  // whether this object should be configured for gradient computation

  // state variables for predictive
  std::vector<double> points_to_sample;  // points to make predictions about

  // derived variables for predictive component
  std::vector<double> K_star_T;
  std::vector<double> grad_K_star;
  std::vector<double> V;
  std::vector<double> K_inv_times_K_star_T;
  std::vector<double> grad_cov;
  std::vector<double> grad_mix_cov;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PointsToSampleState);
};

// forward declaration to use ExpectedImprovement State pointers/typdefs/refs in Evaluator classes
struct ExpectedImprovementState;
struct OnePotentialSampleExpectedImprovementState;

/**\rst
  A class to encapsulate the computation of expected improvement and its spatial gradient.  It designed to work with any
  gaussian process, so its members are very few.  Additionally, this class has no state and within the context of EI
  optimization, it is meant to be accessed by const reference.

  The random numbers needed for EI computation will be passed as parameters instead of contained as members to make
  multithreading more straightforward.
\endrst*/
class ExpectedImprovementEvaluator final {
 public:
  using StateType = ExpectedImprovementState;
  /**\rst
    Constructs a ExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
        gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the underlying GP
        num_mc_iterations: number of monte carlo iterations
        best_so_far: best (minimum) objective function value (in points_sampled_value)
  \endrst*/
  ExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, int num_mc_iterations, double best_so_far) : dim_(gaussian_process_in.dim()), num_mc_iterations_(num_mc_iterations), best_so_far_(best_so_far), gaussian_process_(&gaussian_process_in) {
  }

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  const GaussianProcess * gaussian_process() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return gaussian_process_;
  }

  /**\rst
    Wrapper for ComputeExpectedImprovement(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeExpectedImprovement(ei_state);
  }

  /**\rst
    Wrapper for ComputeGradExpectedImprovement(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * ei_state, double * restrict grad_EI) const OL_NONNULL_POINTERS {
    ComputeGradExpectedImprovement(ei_state, grad_EI);
  }

  /**\rst
    Computes the expected improvement EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+], where Xs are potential points to sample
    and X are already sampled points.  The ^+ indicates that the expression in the expectation evaluates to 0 if it is
    negative.  f^*(X) is the MINIMUM over all known function evaluations (points_sampled_value), whereas f(Xs) are GP-predicted
    function evaluations.

    The EI is the expected improvement in the current best known objective function value that would result from sampling
    at points_to_sample.

    In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.

    \param
        ei_state[1]: properly configured state object

    \return
        ei_state[1]: state with temporary storage modified; normal_rng modified
        the expected improvement from sampling points_to_sample
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /**\rst
    Computes the (partial) derivatives of the expected improvement with respect to each component of the
    index_of_current_point-th point in points_to_sample

    In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
    Monte-Carlo simulation to approximate it.

    \param
        ei_state[1]: properly configured state object

    \return
        ei_state[1]: state with temporary storage modified; normal_rng modified
        grad_EI[dim]: gradient of expected improvement wrt each dimension of the index_of_current_point-th entry in points_to_sample
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(ExpectedImprovementEvaluator);

 private:
  const int dim_;  // spatial dimension (e.g., entries per point of points_sampled)
  int num_mc_iterations_;  // number of monte carlo iterations
  double best_so_far_;  // best (minimum) objective function value (in points_sampled_value)
  const GaussianProcess * gaussian_process_;  // pointer to gaussian process used in EI computations
};

/**\rst
  State object for ExpectedImprovementEvaluator.  This tracks the current set of potential samples (aka points_to_sample) ALONG
  with the current point being evaluated via expected improvement (called current_point).  Current point joined with
  points_to_sample is stored in union_of_points; current_point is assumed to be placed at index = kIndexOfCurrentPoint.

  This struct also tracks the state of the GaussianProcess that underlies the expected improvement computation: the GP state
  is built to handle the initial union_of_points, and subsequent updates to current_point in this object also update the GP state.

  This struct also holds a pointer to a random number generator needed for Monte Carlo integrated EI computations.
  Users MUST guarantee that multiple state objects DO NOT point to the same RNG (in a multithreaded env).

  See general comments on State structs in gpp_common.hpp's header docs.
\endrst*/
struct ExpectedImprovementState final {
  using EvaluatorType = ExpectedImprovementEvaluator;

  static constexpr int kIndexOfCurrentPoint = 0;  // position of current_point in union_of_points
  // this is index (0..num_to_sample-1) of points_to_sample to differentiate against; choose 0 by convention

  /**\rst
    Constructs an ExpectedImprovementState object with a specified source of randomness for the purpose of computing EI
    (and its gradient) over the specified set of points to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ei_evaluator).

    .. WARNING::
        This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :points_to_sample[dim][num_to_sample]: list of potential concurrent samples (i.e., test points for GP predictions)
        :num_to_sample: number of potential samples
        :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
        :normal_rng[1]: pointer to a properly initialized* NormalRNG object

    .. WARNING::
        The NormalRNG object must already be seeded.  If multithreaded computation is used for EI, then every state object
        must have a different NormalRNG (different seeds, not just different objects).
  \endrst*/
  ExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, int num_to_sample_in, bool configure_for_gradients_in, NormalRNG * normal_rng_in) OL_NONNULL_POINTERS :
      dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      configure_for_gradients(configure_for_gradients_in),
      union_of_points(points_to_sample, points_to_sample + dim*num_to_sample),
      points_to_sample_state(*ei_evaluator.gaussian_process(), points_to_sample, num_to_sample, configure_for_gradients),
      normal_rng(normal_rng_in),
      to_sample_mean(num_to_sample),
      grad_mu(dim*num_to_sample),
      cholesky_to_sample_var(Square(num_to_sample)),
      grad_chol_decomp(dim*Square(num_to_sample)),
      EI_this_step_from_var(num_to_sample),
      aggregate(dim),
      normals(num_to_sample) {
  }

  ExpectedImprovementState(ExpectedImprovementState&& OL_UNUSED(other)) = default;

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  /**\rst
    Get current point--potential sample whose EI is being evaluated

    \return
    current_point[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict current_point) const noexcept OL_NONNULL_POINTERS {
    std::copy(union_of_points.data() + kIndexOfCurrentPoint*dim, union_of_points.data() + (kIndexOfCurrentPoint+1)*dim, current_point);
  }

  /**\rst
    Change the current location of the potential sample whose EI is being evaluated.
    Update the state's derived quantities to be consistent with the new point.

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :current_point[dim]: coordinates of new current_point
  \endrst*/
  void UpdateCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict current_point) OL_NONNULL_POINTERS {
    // update current point in union_of_points
    std::copy(current_point, current_point + dim, union_of_points.data() + kIndexOfCurrentPoint*dim);

    // evaluate derived quantities for the GP
    points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), union_of_points.data(), num_to_sample, configure_for_gradients);
  }

  /**\rst
    Configures this state object with a new current point, the location of the potential sample whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.

    .. WARNING::
        This object's state is INVALIDATED if the ei_evaluator (including the GaussianProcess it depends on) used in SetupState is mutated!
        SetupState() should be called again in such a situation.

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :current_point[dim]: current point (ei evaluation location) to change to
  \endrst*/
  void SetupState(const EvaluatorType& ei_evaluator, double const * restrict current_point) OL_NONNULL_POINTERS {
    if (unlikely(dim != ei_evaluator.dim())) {
      OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
    }

    // update quantities derived from current_point
    UpdateCurrentPoint(ei_evaluator, current_point);
  }

  // size information
  const int dim;  // spatial dimension (e.g., entries per point of points_sampled)
  const int num_to_sample;  // number of points being sampled concurrently

  const bool configure_for_gradients;  // whether this object should be configured for gradient computation

  // points currently being sampled
  std::vector<double> union_of_points;

  // gaussian process state
  GaussianProcess::StateType points_to_sample_state;

  // random number generator
  NormalRNG * normal_rng;

  // temporary storage: preallocated space used by ExpectedImprovementEvaluator's member functions
  std::vector<double> to_sample_mean;
  std::vector<double> grad_mu;
  std::vector<double> cholesky_to_sample_var;
  std::vector<double> grad_chol_decomp;

  std::vector<double> EI_this_step_from_var;
  std::vector<double> aggregate;
  std::vector<double> normals;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(ExpectedImprovementState);
};

/**\rst
  This is a specialization of the ExpectedImprovementEvaluator class for when the number of potential samples is 1; i.e.,
  num_to_sample == 1.  In this case, we have analytic formulas for computing EI and its gradient.

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

  /**\rst
    Constructs a OnePotentialSampleExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.

    \param
        :gaussian_process: gaussianprocess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
          underlying gp
        :best_so_far: best (minimum) objective function value (in points_sampled_value)
  \endrst*/
  onepotentialsampleexpectedimprovementevaluator(const gaussianprocess& gaussian_process_in, double best_so_far) : dim_(gaussian_process_in.dim()), best_so_far_(best_so_far), normal_(0.0, 1.0), gaussian_process_(&gaussian_process_in) {
  }

  int dim() const noexcept ol_pure_function ol_warn_unused_result {
    return dim_;
  }

  const gaussianprocess * gaussian_process() const noexcept ol_pure_function ol_warn_unused_result {
    return gaussian_process_;
  }

  /**\rst
    wrapper for computeexpectedimprovement(); see that function for details.
  \endrst*/
  double computeobjectivefunction(statetype * ei_state) const ol_nonnull_pointers ol_warn_unused_result {
    return computeexpectedimprovement(ei_state);
  }

  /**\rst
    wrapper for computegradexpectedimprovement(); see that function for details.
  \endrst*/
  void computegradobjectivefunction(statetype * ei_state, double * restrict grad_ei) const ol_nonnull_pointers {
    computegradexpectedimprovement(ei_state, grad_ei);
  }

  /**\rst
    computes the expected improvement ei(xs) = e_n[[f^*_n(x) - min(f(xs_1),...,f(xs_m))]^+]

    uses analytic formulas to evaluate the expected improvement.

    \param
        :ei_state[1]: properly configured state object

    \return
        ei_state[1]: state with temporary storage modified
        the expected improvement from sampling points_to_sample
  \endrst*/
  double ComputeExpectedImprovement(StateType * ei_state) const;

  /**\rst
    Computes the (partial) derivatives of the expected improvement with respect to the point to sample.

    Uses analytic formulas to evaluate the spatial gradient of the expected improvement.

    \param
        :ei_state[1]: properly configured state object

    \return
        ei_state[1]: state with temporary storage modified
        grad_EI[dim]: gradient of expected improvement wrt each dimension of the index_of_current_point-th entry in points_to_sample
  \endrst*/
  void ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(OnePotentialSampleExpectedImprovementEvaluator);

 private:
  const int dim_;  // spatial dimension (e.g., entries per point of points_sampled)
  double best_so_far_;  // best (minimum) objective function value (in points_sampled_value)

  const boost::math::normal_distribution<double> normal_;  // normal distribution object
  const GaussianProcess * gaussian_process_;  // pointer to gaussian process used in EI computations
};

/**\rst
  State object for OnePotentialSampleExpectedImprovementEvaluator.  This tracks the current point being evaluated via
  expected improvement.

  This is just a special case of ExpectedImprovementState; see those class docs for more details.
  See general comments on State structs in gpp_common.hpp's header docs.
\endrst*/
struct OnePotentialSampleExpectedImprovementState final {
  using EvaluatorType = OnePotentialSampleExpectedImprovementEvaluator;

  static constexpr int kIndexOfCurrentPoint = 0;  // position of current_point in union_of_points; there is only 1 point to sample so this must be 0

  /**\rst
    Constructs an OnePotentialSampleExpectedImprovementState object for the purpose of computing EI
    (and its gradient) over the specified point to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ei_evaluator).

    .. WARNING::
        This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.

    .. NOTE::
        num_to_sample = 1 by definition for this case (hence the sizing on grad_mu and grad_chol)

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :points_to_sample[dim][num_to_sample]: list of potential concurrent samples (i.e., test points for GP predictions)
        :num_to_sample: number of potential samples
        :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
        :normal_rng[1]: UNUSED (here to stay consistent with ExpectedImprovementState ctor)
  \endrst*/
  OnePotentialSampleExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, int num_to_sample_in, bool configure_for_gradients_in, NormalRNG * OL_UNUSED(normal_rng_in)) :
      dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      configure_for_gradients(configure_for_gradients_in),
      current_point(points_to_sample, points_to_sample + dim),
      points_to_sample_state(*ei_evaluator.gaussian_process(), points_to_sample, num_to_sample, configure_for_gradients_in),
      grad_mu(dim),
      grad_chol_decomp(dim) {
    if (unlikely(num_to_sample != 1)) {
      OL_THROW_EXCEPTION(InvalidValueException<int>, "num_to_sample MUST be 1!", num_to_sample, 1);
    }
  }

  OnePotentialSampleExpectedImprovementState(OnePotentialSampleExpectedImprovementState&& OL_UNUSED(other)) = default;

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  /**\rst
    Get current point--potential sample whose EI is being evaluated

    \return
    current_point[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict current_point_out) const noexcept OL_NONNULL_POINTERS {
    std::copy(current_point.begin(), current_point.end(), current_point_out);
  }

  /**\rst
    Change the current location of the potential sample whose EI is being evaluated.
    Update the state's derived quantities to be consistent with the new point.

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :current_point[dim]: coordinates of new current_point
  \endrst*/
  void UpdateCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict current_point_in) OL_NONNULL_POINTERS {
    // update current point in union_of_points
    std::copy(current_point_in, current_point_in + dim, current_point.data());

    // evaluate derived quantities
    points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), current_point.data(), num_to_sample, configure_for_gradients);
  }

  /**\rst
    Configures this state object with a new current point, the location of the potential sample whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.

    .. WARNING::
        This object's state is INVALIDATED if the ei_evaluator (including the GaussianProcess it depends on) used in SetupState is mutated!
        SetupState() should be called again in such a situation.

    \param
        :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
        :current_point[dim]: current point (ei evaluation location) to change to
  \endrst*/
  void SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) OL_NONNULL_POINTERS {
    if (unlikely(dim != ei_evaluator.dim())) {
      OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
    }

    UpdateCurrentPoint(ei_evaluator, points_to_sample);
  }

  // size information
  const int dim;  // spatial dimension (e.g., entries per point of points_sampled)
  const int num_to_sample;  // must be 1

  const bool configure_for_gradients;  // whether this object should be configured for gradient computation

  // points currently being sampled
  std::vector<double> current_point;

  // gaussian process state
  GaussianProcess::StateType points_to_sample_state;

  // temporary storage: preallocated space used by OnePotentialSampleExpectedImprovementEvaluator's member functions
  std::vector<double> grad_mu;
  std::vector<double> grad_chol_decomp;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(OnePotentialSampleExpectedImprovementState);
};

/**\rst
  Set up state vector.

  This is a utility function just for reducing code duplication.

  dim is the spatial dimension, ei_evaluator.dim()
  \param
      :ei_evaluator: evaluator object associated w/the state objects being constructed
      :starting_point[dim]: initial point to load into state (must be a valid point for the problem)
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
      :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)

  \return
      state_vector[max_num_threads]: vector of states containing max_num_threads properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupExpectedImprovementState(const OnePotentialSampleExpectedImprovementEvaluator& ei_evaluator, double const * restrict starting_point, int max_num_threads, bool configure_for_gradients, std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(ei_evaluator, starting_point, 1, configure_for_gradients, nullptr);
  }
}

/**\rst
  Set up state vector.

  This is a utility function just for reducing code duplication.

  TODO(eliu): this is pretty similar to the version directly above it for OnePotentialSampleExpectedImprovementEvaluator.
  I could merge them and use template-fu to pick the execution path (at compile-time), e.g.,
  ::

      template <typename ExpectedImprovementEvaluator>
      void SetupExpectedImprovementState(const ExpectedImprovementEvaluator& ei_evaluator, ...) {
        for (...) {
          if (std::is_same<ExpectedImprovementEvaluator, OnePotentialSampleExpectedImprovementEvaluator>::value) {
            state_vector->emplace_back(ei_evaluator, starting_point, 1, configure_for_gradients, nullptr);
          } else {
            state_vector->emplace_back(ei_evaluator, union_of_points.data(), num_to_sample+1, configure_for_gradients, normal_rng + i);
          }
        }
      }
  `is_same<>::value` resolves to 'true' or 'false' at compile-time, so the compiler will ditch the unused paths.  I'm not
  sure if there's a nicer way to template-fu this.

  \param
      :ei_evaluator: evaluator object associated w/the state objects being constructed
      :starting_point[dim]: initial point to load into state (must be a valid point for the problem)
      :points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      :num_to_sample: number of points being sampled concurrently
      :dim: number of spatial dimensions (size of a point, ei_evaluator.dim())
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
      :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
      :normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration

  \return
      state_vector[max_num_threads]: vector of states containing max_num_threads properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupExpectedImprovementState(const ExpectedImprovementEvaluator& ei_evaluator, double const * restrict starting_point, double const * restrict points_to_sample, int num_to_sample, int dim, int max_num_threads, bool configure_for_gradients, NormalRNG * normal_rng, std::vector<typename ExpectedImprovementEvaluator::StateType> * state_vector) {
  std::vector<double> union_of_points((num_to_sample+1)*dim);
  std::copy(starting_point, starting_point + dim, union_of_points.begin());
  std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.begin() + dim);

  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(ei_evaluator, union_of_points.data(), num_to_sample+1, configure_for_gradients, normal_rng + i);
  }
}

/**\rst
  Optimize the Expected Improvement (to find the next best point to sample).  Optimization is done
  using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from gpp_optimization.hpp.
  Please see that file for details on gradient descent and see gpp_optimization_parameters.hpp for the meanings of
  the GradientDescentParameters.

  This function is just a simple wrapper that sets up the Evaluator's State and calls a general template for restarted GD.

  Currently, recommend that initial coordinate values not differ from their true values by more than about 1 order of
  magnitude.  This means a grid will probably be necessary to give a good enough initial guess.

  The 'dumb' search component is provided through ComputeOptimalPointToSampleViaMultistartGradientDescent() (see below).
  Generally, calling that function should be preferred.  This function is meant for 1) easier testing;
  2) if you really know what you're doing.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  local optima (i.e., the gradient may be substantially nonzero).

  \param
      :ei_evaluator: reference to object that can compute ExpectedImprovement and its spatial gradient
      :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization (e.g., number
        :of iterations, tolerances, learning rate)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :initial_point[dim]: initial guess for gradient descent
      :points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      :num_to_sample: number of points being sampled concurrently
      :normal_rng[1]: a NormalRNG object that provides the (pesudo)random source for MC integration

  \return
      normal_rng[1]: NormalRNG object will have its state changed due to random draws
      next_point[dim]: point yielding the best EI according to gradient descent
\endrst*/
template <typename ExpectedImprovementEvaluator, typename DomainType>
void RestartedGradientDescentEIOptimization(const ExpectedImprovementEvaluator& ei_evaluator, const GradientDescentParameters& optimization_parameters, const DomainType& domain, double const * restrict initial_point, double const * restrict points_to_sample, int num_to_sample, NormalRNG * normal_rng, double * restrict next_point) {
  if (unlikely(optimization_parameters.max_num_restarts <= 0)) {
    return;
  }
  int dim = ei_evaluator.dim();

  OL_VERBOSE_PRINTF("Expected Improvement Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  std::vector<double> union_of_points((num_to_sample + 1)*dim);
  std::copy(initial_point, initial_point + dim, union_of_points.begin() + ExpectedImprovementEvaluator::StateType::kIndexOfCurrentPoint*dim);
  std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.begin() + (ExpectedImprovementEvaluator::StateType::kIndexOfCurrentPoint+1)*dim);

  typename ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_to_sample+1, true, normal_rng);

  GradientDescentOptimizer<ExpectedImprovementEvaluator, DomainType> gd_opt;
  gd_opt.Optimize(ei_evaluator, optimization_parameters, domain, &ei_state);
  ei_state.GetCurrentPoint(next_point);
}

/**\rst
  Performs multistart gradient descent (MGD) to optimize EI.  Starts a GD run from each
  point in start_point_set.  The point corresponding to the optimal EI* is stored in best_next_point.

  This function wraps MultistartOptimizer<>::MultistartOptimize() (see gpp_optimization.hpp), which provides the multistarting
  component. Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  gpp_optimization.hpp. Please see that file for details on gradient descent and see gpp_optimization_parameters.hpp
  for the meanings of the GradientDescentParameters.

  Currently, recommend that initial coordinate values not differ from their true values by more than about 1 order of
  magnitude, so size the domain and num_multistarts accordingly.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  local optima (i.e., the gradient may be substantially nonzero).

  See ComputeOptimalPointToSampleWithRandomStarts() for additional details on MGD and its use for optimizing EI.  Usually that
  endpoint is preferred.

  .. WARNING::
    this function fails ungracefully if NO improvement can be found!  In that case,
    best_next_point will always be the first point in start_point_set.
    found_flag will indicate whether this occured.

  \param
      :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization (e.g., number
        of iterations, tolerances, learning rate)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :start_point_set[dim][num_multistarts]: set of initial guesses for MGD
      :points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      :num_multistarts: number of points in set of initial guesses
      :num_to_sample: number of points being sampled concurrently
      :best_so_far: value of the best sample so far (must be min(points_sampled_value))
      max_int_steps: maximum number of MC iterations
      max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration

  \return
      normal_rng[max_num_threads]: NormalRNG objects will have their state changed due to random draws
      found_flag[1]: true if best_next_point corresponds to a nonzero EI
      best_next_point[dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void ComputeOptimalPointToSampleViaMultistartGradientDescent(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const DomainType& domain, double const * restrict start_point_set, double const * restrict points_to_sample, int num_multistarts, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, NormalRNG * normal_rng, bool * restrict found_flag, double * restrict best_next_point) {
  // set chunk_size; see gpp_common.hpp header comments, item 7
  const int chunk_size = std::max(std::min(4, std::max(1, num_multistarts/max_num_threads)), num_multistarts/(max_num_threads*10));

  if (num_to_sample == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, max_num_threads, true, &ei_state_vector);

    const int dim = gaussian_process.dim();
    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(dim, 0.0, start_point_set);

    GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimization_parameters, domain, start_point_set, num_multistarts, max_num_threads, chunk_size, ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    const int dim = gaussian_process.dim();
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);

    std::vector<typename ExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, start_point_set, points_to_sample, num_to_sample, dim, max_num_threads, true, normal_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(dim, 0.0, start_point_set);

    GradientDescentOptimizer<ExpectedImprovementEvaluator, DomainType> gd_opt;
    MultistartOptimizer<GradientDescentOptimizer<ExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(gd_opt, ei_evaluator, optimization_parameters, domain, start_point_set, num_multistarts, max_num_threads, chunk_size, ei_state_vector.data(), nullptr, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/**\rst
  Performs multistart (restarted) gradient descent (MGD) to optimize Expected Improvement (EI) starting from num_multistarts
  points selected randomly from the within th domain.
  The point corresponding to the optimal EI* is stored in best_next_point.

  This is the primary endpoint for expected improvement optimization using gradient descent.  It produces an uniform random
  sampling of initial guesses and wraps a series of calls:
  The heart of MGD is in ComputeOptimalPointToSampleViaMultistartGradientDescent(), which wraps
  MultistartOptimizer<...>::MultistartOptimize(...) (in gpp_optimization.hpp).
  The heart of restarted GD is in GradientDescentOptimizer<...>::Optimize() (in gpp_optimization.hpp).
  EI is computed in ComputeExpectedImprovement() and its gradient is in ComputeGradExpectedImprovement(), which are member
  functions of ExpectedImprovementEvaluator and OnePotentialSampleExpectedImprovementEvaluator.

  This function should be the primary entry-point into this optimal learning library.  It takes in a covariance function,
  points already sampled, and potential (future) points to sample as well as parameters for gradient descent and MC integration.
  Basically, the input is a full specification of the optimization problem.
  And the output is the "best" next point to sample.

  See file docs for this file for more high level description of how this works.  See the matching cpp file docs
  for more mathematical details.

  Currently, recommend that initial coordinate values not differ from their true values by more than about 1 order of
  magnitude, so size the domain and num_multistarts accordingly.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  local optima (i.e., the gradient may be substantially nonzero).

  .. WARNING::
    this function fails if NO improvement can be found!  In that case,
    best_next_point will always be the first randomly chosen point.
    found_flag will be set to false in this case.

  \param
      :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      :optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization (e.g., number
        of iterations, tolerances, learning rate)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      :num_to_sample: number of points being sampled concurrently
      :best_so_far: value of the best sample so far (must be min(points_sampled_value))
      :max_int_steps: maximum number of MC iterations
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration

  \return
      found_flag[1]: true if best_next_point corresponds to a nonzero EI
      uniform_generator[1]:UniformRandomGenerator object will have its state changed due to random draws
      normal_rng[max_num_threads]: NormalRNG objects will have their state changed due to random draws
      best_next_point[dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void ComputeOptimalPointToSampleWithRandomStarts(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const DomainType& domain, double const * restrict points_to_sample, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_next_point) {
  std::vector<double> starting_points(gaussian_process.dim()*optimization_parameters.num_multistarts);

  // GenerateUniformPointsInDomain() is allowed to return fewer than the requested number of multistarts
  int num_multistarts = domain.GenerateUniformPointsInDomain(optimization_parameters.num_multistarts, uniform_generator, starting_points.data());

  ComputeOptimalPointToSampleViaMultistartGradientDescent(gaussian_process, optimization_parameters, domain, starting_points.data(), points_to_sample, num_multistarts, num_to_sample, best_so_far, max_int_steps, max_num_threads, normal_rng, found_flag, best_next_point);
#ifdef OL_WARNING_PRINT
  if (false == *found_flag) {
    OL_WARNING_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
    OL_WARNING_PRINTF("First multistart point was returned:\n");
    PrintMatrix(starting_points.data(), 1, gaussian_process.dim());
  }
#endif
}

/**\rst
  Function to evaluate Expected Improvement over a specified list of num_multistarts points.
  Optionally outputs the EI at each of these points.
  Outputs the point of the set obtaining the maximum EI value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
  This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

  This function is just a wrapper that builds the required state objects and a NullOptimizer object and calls
  MultistartOptimizer<...>::MultistartOptimize(...); see gpp_optimization.hpp.

  \param
      gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      initial_guesses[dim][num_multistarts]: list of points at which to compute EI
      points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      num_multistarts: number of points to check
      num_to_sample: number of points being sampled concurrently
      best_so_far: value of the best sample so far (must be min(points_sampled_value))
      max_int_steps: maximum number of MC iterations
      max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration

  \return
      found_flag[1]: true if best_next_point corresponds to a nonzero EI
      normal_rng[max_num_threads]: NormalRNG objects will have their state changed due to random draws
      function_values[num_multistarts]: EI evaluated at each point of initial_guesses, in the same order as initial_guesses; never dereferenced if nullptr
      best_next_point[dim]: point yielding the best EI according to dumb search
\endrst*/
template <typename DomainType>
void EvaluateEIAtPointList(const GaussianProcess& gaussian_process, const DomainType& domain, double const * restrict initial_guesses, double const * restrict points_to_sample, int num_multistarts, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool *  restrict found_flag, NormalRNG * normal_rng, double * restrict function_values, double * restrict best_next_point) {
  // set chunk_size; see gpp_common.hpp header comments, item 7
  const int chunk_size = std::max(std::min(40, std::max(1, num_multistarts/max_num_threads)), num_multistarts/(max_num_threads*120));

  if (num_to_sample == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, max_num_threads, false, &ei_state_vector);

    const int dim = gaussian_process.dim();
    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(dim, 0.0, initial_guesses);

    NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, domain, initial_guesses, num_multistarts, max_num_threads, chunk_size, ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    const int dim = gaussian_process.dim();
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);

    std::vector<typename ExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, points_to_sample, num_to_sample, dim, max_num_threads, false, normal_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(dim, 0.0, initial_guesses);

    NullOptimizer<ExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<ExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<ExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, domain, initial_guesses, num_multistarts, max_num_threads, chunk_size, ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/**\rst
  Function to do a "dumb" search over num_multistarts points (generated on a Latin Hypercube) best point to sample
  next (i.e., point with the largest Expected Improvement).

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  local optima (i.e., the gradient may be substantially nonzero).

  Wraps EvaluateEIAtPointList(); constructs the input point list with a uniform random sampling from the given Domain object.

  \param
      gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      num_multistarts: number of random points to check
      num_to_sample: number of points being sampled concurrently
      best_so_far: value of the best sample so far (must be min(points_sampled_value))
      max_int_steps: maximum number of MC iterations
      max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration

  \return
      found_flag[1]: true if best_next_point corresponds to a nonzero EI
      uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      normal_rng[max_num_threads]: NormalRNG objects will have their state changed due to random draws
      next_point_winner[dim]: point yielding the best EI according to dumb search
\endrst*/
template <typename DomainType>
void ComputeOptimalPointToSampleViaLatinHypercubeSearch(const GaussianProcess& gaussian_process, const DomainType& domain, double const * restrict points_to_sample, int num_multistarts, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts);
  num_multistarts = domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator, initial_guesses.data());

  EvaluateEIAtPointList(gaussian_process, domain, initial_guesses.data(), points_to_sample, num_multistarts, num_to_sample, best_so_far, max_int_steps, max_num_threads, found_flag, normal_rng, nullptr, best_next_point);
}

/**\rst
  Returns the optimal set of points to sample CONCURRENTLY.  That is, we may want to run 4 experiments at the same time and
  maximize the EI across all 4 experiments at once.   This function handles this use case.

  With lhc_search_only := false and num_samples_to_generate := 1, this is equivalent to
  ComputeOptimalPointToSampleWithRandomStarts().

  .. NOTE..
    In **Parameters**, note the difference between points_to_sample/num_to_sample and num_samples_to_generate.
    points_to_sample are experiments that are ALREADY ongoing.  num_samples_to_generate tells this function how many NEW
    sample points to return.

  \param
      gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
        underlying GP
      optimization_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization (e.g., number
        of iterations, tolerances, learning rate)
      domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      points_to_sample[dim][num_to_sample]: points that are being sampled concurrently from the GP
      num_to_sample: number of points being sampled concurrently
      best_so_far: value of the best sample so far (must be min(points_sampled_value))
      max_int_steps: maximum number of MC iterations
      max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
      num_lhc_samples: number of samples to draw if/when doing latin hypercube search
      num_samples_to_generate: how many simultaneous experiments you would like to run
      uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration
  \return
      found_flag[1]: true if best_points_to_sample corresponds to a nonzero EI if sampled simultaneously
      uniform_generator[1]:UniformRandomGenerator object will have its state changed due to random draws
      normal_rng[max_num_threads]: NormalRNG objects will have their state changed due to random draws
      best_points_to_sample[num_samples_to_generate*dim]: point yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const DomainType& domain, double const * restrict points_to_sample, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const TensorProductDomain& domain, double const * restrict points_to_sample, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);
extern template void ComputeOptimalSetOfPointsToSample(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimization_parameters, const SimplexIntersectTensorProductDomain& domain, double const * restrict points_to_sample, int num_to_sample, double best_so_far, int max_int_steps, int max_num_threads, bool lhc_search_only, int num_lhc_samples, int num_samples_to_generate, bool * restrict found_flag, UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MATH_HPP_
