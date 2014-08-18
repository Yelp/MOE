/*!
  \file gpp_math.cpp
  \rst
  These comments are getting to be of some length, so here's a table of contents:

  1. FILE OVERVIEW
  2. IMPLEMENTATION NOTES
  3. MATHEMATICAL OVERVIEW

     a. GAUSSIAN PROCESSES
     b. SAMPLING FROM GPs
     c. EXPECTED IMPROVEMENT

  4. CODE DESIGN/LAYOUT OVERVIEW:

     a. class GaussianProcess
     b. class ExpectedImprovementEvaluator, OnePotentialSampleExpectedImprovementEvaluator
     c. function ComputeOptimalPointsToSampleWithRandomStarts()

  5. CODE HIERARCHY / CALL-TREE

  **1. FILE OVERVIEW**

  Implementations of functions for Gaussian Processes (mean, variance of GPs and their gradients) and for
  computing and optimizing Expected Improvement (EI).

  **2. IMPLEMENTATION NOTES**

  See gpp_math.hpp file docs and gpp_common.hpp for a few important implementation notes
  (e.g., restrict, memory allocation, matrix storage style, etc), as well as citation details.

  Additionally, the matrix looping idioms used in this file deserve further mention: see gpp_common.hpp
  header comments, item 7 for further details.  In summary, using matrix-vector-multiply as an example, we do::

    for (int i = 0; i < m; ++i) {
      y[i] = 0;
      for (int j = 0; j < n; ++j) {
        y[i] += A[j]*x[j];
      }
      A += n;
    }

  **3. MATHEMATICAL OVERVIEW**

  Next, we provide a high-level discussion of GPs and the EI optimization process used in this file.  See
  Rasmussen & Williams for more details on the former and Scott Clark's thesis for details on the latter.  This segment
  is more focused on concepts and mathematical ideas.  We subsequently discuss how the classes and functions
  in this file map onto these mathematical concepts.  If it wasn't clear, please read the file comments for
  gpp_math.hpp before continuing (a conceptual overview).

  **3a. GAUSSIAN PROCESSES**

  First, a Gaussian Process (GP) is defined as a collection of normally distributed random variables (RVs); these
  RVs are not independent nor identically-distributed (i.e., all normal but different mean/var) in general.  Since
  the GP is a collection of RVs, it defines a distribution over FUNCTIONS.  So drawing from the GP realizes
  one particular function.

  Now let X = training data; these are our experimental independent variables
  let f = training data observed values; this is our (SCALAR) dependent-variable
  So for ``(X_i, f_i)`` pairs, we say:

  ``f ~ GP(0, cov(X,X)) /equiv N(0, cov(X,X))``

  the training data, f, is distributed like a (multi-variate) Gaussian with mean 0 and ``variance = cov(X,X)``.
  Drawing from this GP requires conditioning on the result satisfying the training data.  That is, the realized
  function must pass through all points ``(X,f)``.  Between these, "essentially any" behavior is possible, although certain
  behaviors are more likely as specified via ``cov(X,X)``.
  Note that the GP has 0 mean (and no signal variance) to specify that it passes through X,f exactly.  Nonzero mean
  would shift the entire distribution so that it passes through ``(X,f+mu)``.

  In the following, K(X,X) is the covariance function.  It's given as an input to this whole process and is critical
  in informing the behavior of the GP.  The covariance function describes how related we (a priori) believe prior
  points are to each other.
  In code, the covariance function is specified through the CovarianceInterface class.

  In a noise-free setting (signal noise modifies ``K`` to become ``K + \sigma^2 * Id``, ``Id`` being identity), the joint
  distribution of training inputs, ``f``, and test outputs, ``fs``, is::

    [ f  ]  ~ N( 0, [ K(X,X)   K(X,Xs)  ]  = [ K     Ks  ]         (Equation 1, Rasmussen & Williams 2.18)
    [ fs ]          [ K(Xs,X)  K(Xs,Xs) ]    [ Ks^T  Kss ]

  where the test outputs are drawn from the prior.

  | ``K(X,X)`` and ``K(Xs,Xs)`` are computed in BuildCovarianceMatrix()
  | ``K(X,Xs)`` is computed by BuildMixCovarianceMatrix(); and ``K(Xs,X)`` is its transpose.
  | ``K + \sigma^2`` is computed in BuildCovarianceMatrixWithNoiseVariance(); almost all practical uses of GPs and EI will

  be over data with nonzero noise variance.  However this is immaterial to the rest of the discussion here.

  **3b. SAMPLING FROM GPs**

  So to obtain the posterior distribution, fs, we again sample this joint prior and throw out any function
  realizations that do not satisfy the observations (i.e., pass through all ``(X,f)`` pairs).  This is expensive.

  Instead, we can use math to compute the posterior by conditioning it on the prior:

  ``fs | Xs,X,f ~ N( mus, Vars)``

  where ``mus = K(Xs,X) * K(X,X)^-1 * f = Ks^T * K^-1 * f,  (Equation 2, Rasmussen & Williams 2.19)``
  which is computed in GaussianProcess::ComputeMeanOfPoints.

  and  ``Vars = K(Xs,Xs) - K(Xs,X) * K(X,X)^-1 * K(X,Xs) = Kss - Ks^T * K^-1 * Ks, (Equation 3, Rasumussen & Williams 2.19)``
  which is implemented in GaussianProcess::ComputeVarianceOfPoints (and provably SPD).

  Now we can draw from this multi-variate Gaussian by:

  ``y = mus + L * w    (Equation 4)``

  where ``L * L^T = Vars`` (cholesky-factorization) and w is a vector of samples from ``N(0,1)``
  Note that if our GP has 10 dimensions (variables), then y contains 10 sample values.

  **3c. EXPECTED IMPROVEMENT**

  .. Note:: these comments are copied in Python: interfaces/expected_improvement_interface.py

  Then the improvement for this single sample is::

    I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5)
        {          0               else

  And the expected improvement, EI, can be computed by averaging repeated computations of I; i.e., monte-carlo integration.
  This is done in ExpectedImprovementEvaluator::ComputeExpectedImprovement(); we can also compute the gradient. This
  computation is needed in the optimization of q,p-EI.

  There is also a special, analytic case of EI computation that does not require monte-carlo integration. This special
  case can only be used to compute 1,0-EI (and its gradient). Still this can be very useful (e.g., the heuristic
  optimization in gpp_heuristic_expected_improvement_optimization.hpp estimates q,0-EI by repeatedly solving
  1,0-EI).

  From there, since EI is taken from a sum of gaussians, we expect it to be reasonably smooth
  and apply multistart, restarted gradient descent to find the optimum.  The use of gradient descent
  implies the need for all of the various "grad" functions, e.g., GP::ComputeGradMeanOfPoints().
  This is handled starting in the highest level functions of file, ComputeOptimalPointsToSample().

  **4. CODE OVERVIEW**

  Finally, we give some further details about how the previous ideas map into the code.  We begin with an overview
  of important classes and functions in this file, and end by going over the call stack for the EI optimization entry point.

  **4a. First, the GaussianProcess (GP) class**

  The GaussianProcess class abstracts the handling of GPs and their properties; quickly going over the functionality: it
  provides methods for computing mean, variance, cholesky of variance, and their gradients (wrt spatial dimensions).
  GP also allows the user to sample function values from it, distributed according to the GP prior.  Lastly GP provides
  the ability to change the hyperparameters of its covariance function (although currently you cannot change the
  covariance function; this would not be difficult to add).

  Computation-wise, GaussianProcess also makes precomputation and preallocation convenient.  The class tracks all of its
  inputs (e.g., ``X``, ``f``, noise var, covariance) as well as quantities that are derivable from *only* these inputs; e.g.,
  ``K``, cholesky factorization of ``K``, ``K^-1*y``.  Thus repeated calculations with the GP over the same training data avoids
  (very expensive) factorizations of ``K``.

  A last note about GP: it uses the State idiom laid out in gpp_common.hpp.  The associated state is PointsToSampleState.
  PointsToSampleState tracks the current "test" data set, points_to_sample--the set of currently running experiments,
  possibly including the current point(s) being optimized. In the q,p-EI terminology, PointsToSampleState tracks the
  union of ``points_to_sample`` and ``points_being_sampled``. PointsToSampleState preallocates all vectors needed by GP's
  member functions; it also precomputes (per ``points_to_sample`` update) some derived quantities that are used repeatedly
  by GP member functions.

  In current usage, users generally will not need to access GaussianProcess's member functions directly; instead these are
  used indirectly when users compute or optimize EI.  Plotting/visualization might be one reason to call GP members directly.

  **4b. Next, the ExpectedImprovementEvaluator and OnePotentialSampleExpectedImprovementEvaulator classes**

  ExpectedImprovementEvaluator abstracts the computation of EI and its gradient.  This class references a single
  GaussianProcess that it uses to compute EI/grad EI as described above.  Equations 4, 5 above detailed the EI computation;
  further details can be found below in the call tree discussion as well as in the implementation docs for these
  functions.  The gradient of EI is implemented similarly; see implementation docs for details on the one subtlety.

  OnePotentialSample is a special case of ExpectedImprovementEvaluator. With ``num_to_sample = 1`` and ``num_being_sampled = 0``
  (only occurs in 1,0-EI evaluation/optimization), there is only one experiment to worry about and no concurrent events.
  This simplifies the EI computation substantially (multi-dimensional Gaussians become a simple one dimensional case)
  and we can write EI analytically in terms of the PDF and CDF of a N(0,1) normal distribution (which are evaluated
  numerically by boost). No monte-carlo necessary!

  ExpectedImprovementEvaluator and OnePotentialSample have corresponding State classes as well.  These are similar
  to each other except OnePotentialSample does not have a NormalRNG pointer (since it does no MC integration) and some
  temporaries are dropped since they have size 1.  But for the general EI's State, the NormalRNG pointer must reference
  a different object for each thread!  Notably, both EI State classes construct their own GaussianProcess::StateType
  object for use with GP members.  As long as there is only one EI state per thread, This ensures thread safety since there
  is never a reason (or a way) for multiple threads to accidentally use the same GP state.  Finally, the EI state classes
  hold some pre-allocated vectors for use as local temporaries by EI and GradEI computation.

  **4c. And finally, we discuss selecting optimal experiments with ComputeOptimalPointsToSampleWithRandomStarts()**

  This function is the top of the hierarchy for EI optimization.  It encompasses a multistart, restarted gradient descent
  method.  Since this is not a convex optimization problem, there could be multiple local optima (or even 0 optima).  So
  we start GD from multiple locations (multistart) as a heuristic in hopes of finding the global optima.

  See the file comments of gpp_optimization.hpp for more details on the base gradient descent implementation and the restart
  component of restarted gradient descent.

  **5. CODE HIERARCHY / CALL-TREE**

  For obtaining multiple new points to sample (q,p-EI), we have two main paths for optimization: multistart gradient
  descent and 'dumb' search. The optimization hierarchy looks like (these optimization functions are in the header;
  they are templates):
  ComputeOptimalPointsToSampleWithRandomStarts<...>(...)  (selects random points; defined in math.hpp)

  * Solves q,p-EI.
  * Selects random starting locations based on random sampling from the domain (e.g., latin hypercube)
  * This calls:

    ComputeOptimalPointsToSampleViaMultistartGradientDescent<...>(...)  (multistart gradient descent)

    * Switches into analytic OnePotentialSample case when appropriate
    * Multithreaded over starting locations
    * Optimizes with restarted gradient descent; collects results and updates the solution as new optima are found
    * This calls:

      MultistartOptimizer<...>::MultistartOptimize(...) for multistarting (see gpp_optimization.hpp) which in turn uses
      GradientDescentOptimizer::Optimize<ObjectiveFunctionEvaluator, Domain>() (see gpp_optimization.hpp)

  ComputeOptimalPointsToSampleViaLatinHypercubeSearch<...>(...)  (defined in gpp_math.hpp)

  * Estimates q,p-EI with a 'dumb' search.
  * Selects random starting locations based on random sampling from the domain (e.g., latin hypercube)
  * This calls:

    EvaluateEIAtPointList<...>(...)

    * Evaluates EI at each starting location
    * Switches into analytic OnePotentialSample case when appropriate
    * Multithreaded over starting locations
    * This calls:

      MultistartOptimizer<...>::MultistartOptimize(...) for multistarting (see gpp_optimization.hpp)

  ComputeOptimalPointsToSample<...>(...)  (defined in gpp_math.cpp)

  * Solves q,p-EI
  * Tries ComputeOptimalPointsToSampleWithRandomStarts() first.
  * If that fails, switches to ComputeOptimalPointsToSampleViaLatinHypercubeSearch().

  So finally we will overview the function calls for EI calculation.  We limit our discussion to the general MC case;
  the analytic case is similar and simpler.
  ExpectedImprovementEvaluator::ComputeExpectedImprovement()  (computes EI)

  * Computes GP.mean, GP.variance, cholesky(GP.variance)
  * MC integration: samples from the GP repeatedly (Equation 4) and computes the improvement (Equation 5), averaging the result
    See function comments for more details.
  * Calls out to GP::ComputeMeanOfPoints(), GP:ComputeVarianceOfPoints, ComputeCholeskyFactorL, NormalRNG::operator(),
    and TriangularMatrixVectorMultiply

  ExpectedImprovementEvaluator::ComputeGradExpectedImprovement()  (computes gradient of EI)

  * Compute GP.mean, variance, cholesky(variance), grad mean, grad variance, grad cholesky variance
  * MC integration: Equation 4, 5 as before to compute improvement each step
    Only have grad EI contributions when improvement > 0.
    Care is needed because only the point yielding the largest improvement contributes to the gradient.
    See function comments for more details.

  We will not detail the call tree once inside of GaussianProcess.  The mathematical formulas for the mean and variance
  were already described above (Equation 2, 3).  Function docs (in this file) further detail/cite the formulas and
  relevant derivations for gradients of these quantities.  Suffice to say there's a lot of linear algebra.  Read on
  (to those fcn docs) for further details but this does little to expose the important concepts behind EI and GP.
\endrst*/

#include "gpp_math.hpp"

#include <cmath>

#include <algorithm>
#include <memory>
#include <vector>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_linear_algebra-inl.hpp"
#include "gpp_logging.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

namespace {  // utilities for A_{k,j,i}*x_j and building covariance matrices

/*!\rst
  Helper function to perform the following math (in index notation)::

    y_{k,i} = A_{k,j,i} * x_j
    0 <= i < dim_one, 0 <= j < dim_two, 0 <= k < dim_three

  This is nothing more than dim_one matrix-vector products ``A_{k,j} * x_j``, and could be implemented using a
  single GeneralMatrixMatrixMultiply if A were stored (full) block diagonal (but this wastes a lot of space).

  \param
    :tensor[dim_three][dim_two][dim_one]: tensor multiplicand
    :vector[dim_two]: vector multiplicand
    :dim_one: first dimension of tensor
    :dim_two: second dimension of tensor
    :dim_three: third dimension of tensor
  \output
    :answer[dim_three][dim_one]: result matrix
\endrst*/
OL_NONNULL_POINTERS void SpecialTensorVectorMultiply(double const * restrict tensor,
                                                     double const * restrict vector,
                                                     int dim_one, int dim_two, int dim_three,
                                                     double * restrict answer) noexcept {
  for (int i = 0; i < dim_one; ++i) {
    GeneralMatrixVectorMultiply(tensor, 'N', vector, 1.0, 0.0, dim_three, dim_two, dim_three, answer);
    tensor += dim_two*dim_three;
    answer += dim_three;
  }
}

/*!\rst
  .. NOTE:: These comments have been copied into build_covariance_matrix in python_version/python_utils.py.

  Compute the covariance matrix, ``K``, of a list of points, ``X_i``.  Matrix is computed as:

  ``A_{i,j} = covariance(X_i, X_j)``.

  Result is SPD assuming covariance operator is SPD and points are unique.

  Generally, this is called from other functions with "points_sampled" as the input and not any
  arbitrary list of points; hence the very specific input name.

  Point list cannot contain duplicates.  Doing so (or providing nearly duplicate points) can lead to
  semi-definite matrices or very poor numerical conditioning.

  \param
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :points_sampled[dim][num_sampled]: list of points
    :dim: spatial dimension of a point
    :num_sampled: number of points
  \output
    :cov_matrix[num_sampled][num_sampled]: computed covariance matrix
\endrst*/
OL_NONNULL_POINTERS void BuildCovarianceMatrix(const CovarianceInterface& covariance,
                                               double const * restrict points_sampled,
                                               int dim, int num_sampled, double * restrict cov_matrix) noexcept {
  // we only work with lower triangular parts of symmetric matrices, so only fill half of it
  for (int i = 0; i < num_sampled; ++i) {
    for (int j = i; j < num_sampled; ++j) {
      cov_matrix[j] = covariance.Covariance(points_sampled + i*dim, points_sampled+j*dim);
    }
    cov_matrix += num_sampled;
  }
}

/*!\rst
  Same as BuildCovarianceMatrix, except noise variance ``(\sigma_n^2)`` is added to the main diagonal.

  Only additional inputs listed; see BuildCovarianceMatrix() for other arguments.

  \param
    :noise_variance[num_sampled]: i-th entry is amt of noise variance to add to i-th diagonal entry; i.e., noise measuring i-th point
\endrst*/
OL_NONNULL_POINTERS void BuildCovarianceMatrixWithNoiseVariance(const CovarianceInterface& covariance,
                                                                double const * restrict noise_variance,
                                                                double const * restrict points_sampled,
                                                                int dim, int num_sampled,
                                                                double * restrict cov_matrix) noexcept {
  // we only work with lower triangular parts of symmetric matrices, so only fill half of it
  for (int i = 0; i < num_sampled; ++i) {
    for (int j = i; j < num_sampled; ++j) {
      cov_matrix[j] = covariance.Covariance(points_sampled + i*dim, points_sampled+j*dim);
    }
    cov_matrix[i] += noise_variance[i];
    cov_matrix += num_sampled;
  }
}

/*!\rst
  .. NOTE:: These comments have been copied into build_mix_covariance_matrix in python_version/python_utils.py.

  Compute the "mix" covariance matrix, ``Ks``, of ``X`` and ``Xs`` (``points_sampled`` and ``points_to_sample``, respectively).
  Matrix is computed as:

  ``A_{i,j} = covariance(X_i, Xs_j).``

  Result is not guaranteed to be SPD and need not even be square.

  Generally, this is called from other functions with "points_sampled" and "points_to_sample" as the
  input lists and not any arbitrary list of points; hence the very specific input name.  But this
  is not a requirement.

  Point lists cannot contain duplicates with each other or within themselves.

  \param
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :points_sampled[dim][num_sampled]: list of points, ``X``
    :points_to_sample[dim][num_to_sample]: list of points, ``Xs``
    :dim: spatial dimension of a point
    :num_sampled: number of points in points_sampled
    :num_to_sample: number of points in points_to_sample
  \output
    :cov_matrix[num_sampled][num_to_sample]: computed "mix" covariance matrix
\endrst*/
OL_NONNULL_POINTERS void BuildMixCovarianceMatrix(const CovarianceInterface& covariance,
                                                  double const * restrict points_sampled,
                                                  double const * restrict points_to_sample,
                                                  int dim, int num_sampled, int num_to_sample,
                                                  double * restrict cov_matrix) noexcept {
  // calculate the covariance matrix defined in gpp_covariance.hpp
  for (int j = 0; j < num_to_sample; ++j) {
    for (int i = 0; i < num_sampled; ++i) {
      cov_matrix[i] = covariance.Covariance(points_sampled + i*dim, points_to_sample + j*dim);
    }
    cov_matrix += num_sampled;
  }
}

}  // end unnamed namespace

void GaussianProcess::BuildCovarianceMatrixWithNoiseVariance() noexcept {
  optimal_learning::BuildCovarianceMatrixWithNoiseVariance(*covariance_ptr_, noise_variance_.data(),
                                                           points_sampled_.data(), dim_, num_sampled_,
                                                           K_chol_.data());
}

void GaussianProcess::BuildMixCovarianceMatrix(double const * restrict points_to_sample,
                                               int num_to_sample,
                                               double * restrict covariance_matrix) const noexcept {
  optimal_learning::BuildMixCovarianceMatrix(*covariance_ptr_, points_sampled_.data(),
                                             points_to_sample, dim_, num_sampled_,
                                             num_to_sample, covariance_matrix);
}

void GaussianProcess::RecomputeDerivedVariables() {
  // resize if needed
  if (unlikely(static_cast<int>(K_inv_y_.size()) != num_sampled_)) {
    K_chol_.resize(num_sampled_*num_sampled_);
    K_inv_y_.resize(num_sampled_);
  }

  // recompute derived quantities
  BuildCovarianceMatrixWithNoiseVariance();
  int leading_minor_index = ComputeCholeskyFactorL(num_sampled_, K_chol_.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException,
                       "Covariance matrix (K) singular. Check for duplicate points_sampled "
                       "(with 0 noise) and/or extreme hyperparameter values.",
                       K_chol_.data(), num_sampled_, leading_minor_index);
  }

  std::copy(points_sampled_value_.begin(), points_sampled_value_.end(), K_inv_y_.begin());
  CholeskyFactorLMatrixVectorSolve(K_chol_.data(), num_sampled_, K_inv_y_.data());
}

GaussianProcess::GaussianProcess(const CovarianceInterface& covariance_in,
                                 double const * restrict points_sampled_in,
                                 double const * restrict points_sampled_value_in,
                                 double const * restrict noise_variance_in,
                                 int dim_in, int num_sampled_in)
    : dim_(dim_in),
      num_sampled_(num_sampled_in),
      covariance_ptr_(covariance_in.Clone()),
      points_sampled_(points_sampled_in, points_sampled_in + num_sampled_in*dim_in),
      points_sampled_value_(points_sampled_value_in, points_sampled_value_in + num_sampled_in),
      noise_variance_(noise_variance_in, noise_variance_in + num_sampled_),
      K_chol_(num_sampled_in*num_sampled_in),
      K_inv_y_(num_sampled_),
      normal_rng_(kDefaultSeed) {
  RecomputeDerivedVariables();
}

GaussianProcess::GaussianProcess(const GaussianProcess& source)
    : dim_(source.dim_),
      num_sampled_(source.num_sampled_),
      covariance_ptr_(source.covariance_ptr_->Clone()),
      points_sampled_(source.points_sampled_),
      points_sampled_value_(source.points_sampled_value_),
      noise_variance_(source.noise_variance_),
      K_chol_(source.K_chol_),
      K_inv_y_(source.K_inv_y_),
      normal_rng_(source.normal_rng_) {
}

/*!\rst
  Sets up precomputed quantities needed for mean, variance, and gradients thereof.  These quantities are:

  ``Ks := Ks_{k,i} = cov(X_k, Xs_i)`` (used by mean, variance)

  Then if we need gradients:

  | ``K^-1 * Ks := solution X of K_{k,l} * X_{l,i} = Ks{k,i}`` (used by variance, grad variance)
  | ``gradient of Ks := C_{d,k,i} = \pderiv{Ks_{k,i}}{Xs_{d,i}}`` (used by grad mean, grad variance)
\endrst*/
void GaussianProcess::FillPointsToSampleState(StateType * points_to_sample_state) const {
  BuildMixCovarianceMatrix(points_to_sample_state->points_to_sample.data(),
                           points_to_sample_state->num_to_sample, points_to_sample_state->K_star.data());

  if (points_to_sample_state->num_derivatives > 0) {
    // to save on duplicate storage, precompute K^-1 * Ks
    std::copy(points_to_sample_state->K_star.begin(), points_to_sample_state->K_star.end(),
              points_to_sample_state->K_inv_times_K_star.begin());
    CholeskyFactorLMatrixMatrixSolve(K_chol_.data(), num_sampled_, points_to_sample_state->num_to_sample,
                                     points_to_sample_state->K_inv_times_K_star.data());

    double * restrict gKs_temp = points_to_sample_state->grad_K_star.data();
    // also precompute C_{d,k,i} = \pderiv{Ks_{k,i}}{Xs_{d,i}}, stored in grad_K_star_
    for (int i = 0; i < points_to_sample_state->num_derivatives; ++i) {
      for (int j = 0; j < num_sampled_; ++j) {
        covariance_ptr_->GradCovariance(points_to_sample_state->points_to_sample.data() + i*dim_,
                                        points_sampled_.data() + j*dim_, gKs_temp);
        gKs_temp += dim_;
      }
    }
  }
}

/*!\rst
  Calculates the mean (from the GPP) of a set of points:

  ``mus = Ks^T * K^-1 * y``

  See Rasmussen and Willians page 19 alg 2.1
\endrst*/
void GaussianProcess::ComputeMeanOfPoints(const StateType& points_to_sample_state,
                                          double * restrict mean_of_points) const noexcept {
  GeneralMatrixVectorMultiply(points_to_sample_state.K_star.data(), 'T', K_inv_y_.data(),
                              1.0, 0.0, num_sampled_, points_to_sample_state.num_to_sample, num_sampled_, mean_of_points);
}

/*!\rst
  Gradient of the mean of a GP.  Note that the output storage skips known zeros (see declaration docs for details).
  See Scott Clark's PhD thesis for more spelled out mathematical details, but this is a reasonably straightforward
  differentiation of:

  ``mus = Ks^T * K^-1 * y``

  wrt ``Xs`` (so only Ks contributes derivative terms)
\endrst*/
void GaussianProcess::ComputeGradMeanOfPoints(const StateType& points_to_sample_state,
                                              double * restrict grad_mu) const noexcept {
  SpecialTensorVectorMultiply(points_to_sample_state.grad_K_star.data(), K_inv_y_.data(),
                              points_to_sample_state.num_derivatives, num_sampled_, dim_, grad_mu);
}

/*!\rst
  Mathematically, we are computing Vars (Var_star), the GP variance.  Vars is defined at the top of this file (Equation 3)
  and in Rasmussen & Williams, Equation 2.19:

  | ``L * L^T = K``
  | ``V = L^-1 * Ks``
  | ``Vars = Kss - (V^T * V)``

  This quantity is:

  ``Kss``: the covariance between test points based on the prior distribution

  minus

  ``V^T * V``: the information observations give us about the objective function

  Notice that Vars is clearly symmetric.  ``Kss`` is SPD. And
  ``V^T * V = (V^T * V)^T`` is symmetric (and is in fact SPD).

  ``V^T * V = Ks^T * K^-1 * K_s`` is SPD because:

  ``X^T * A * X`` is SPD when A is SPD AND ``X`` has full rank (``X`` need not be square)

  ``Ks`` has full rank as long as ``K`` & ``Kss`` are SPD; ``K^-1`` is SPD because ``K`` is SPD.

  It turns out that ``Vars`` is SPD.

  In Equation 1 (Rasmussen & Williams 2.18), it is clear that the combined covariance matrix
  is SPD (as long as no duplicate points and the covariance function is valid).  A matrix of the form::

    [ A   B ]
    [ B^T C ]

  is SPD if and only if ``A`` is SPD AND ``(C - B^T * A^-1 * B)`` is SPD.  Here, ``A = K, B = Ks, C = Kss``.
  This (aka Schur Complement) can be shown readily::

    [ A   B ] = [  I            0 ] * [  A    0                ] * [ I   A^-1 * B ]
    [ B^T C ]   [ (A^-1 * B)^T  I ] * [  0 (C - B^T * A^-1 * B)]   [ 0       I    ]

  This factorization is valid because ``A`` is SPD (and thus invertible).  Then by the ``X^T * A * X`` rule for SPD-ness,
  we know the block-diagonal matrix in the center is SPD.  Hence the SPD-ness of ``V^T * V`` follows readily.

  For more information, see:
  http://en.wikipedia.org/wiki/Schur_complement
\endrst*/
void GaussianProcess::ComputeVarianceOfPoints(StateType * points_to_sample_state,
                                              double * restrict var_star) const noexcept {
  // optimized code that avoids formation of K_inv
  const int num_to_sample = points_to_sample_state->num_to_sample;

  // Vars = Kss
  BuildCovarianceMatrix(*covariance_ptr_, points_to_sample_state->points_to_sample.data(), dim_, num_to_sample, var_star);
  // following block computes Vars -= V^T*V, with the exact method depending on what quantities were precomputed
  if (unlikely(points_to_sample_state->num_derivatives == 0)) {
    std::copy(points_to_sample_state->K_star.begin(), points_to_sample_state->K_star.end(),
              points_to_sample_state->V.begin());

    // V := L^-1 * K_star
    TriangularMatrixMatrixSolve(K_chol_.data(), 'N', num_sampled_, num_to_sample, num_sampled_,
                                points_to_sample_state->V.data());

    // compute V^T V = (L^-1 * Ks)^T * (L^-1 * Ks).
    GeneralMatrixMatrixMultiply(points_to_sample_state->V.data(), 'T', points_to_sample_state->V.data(),
                                -1.0, 1.0, num_to_sample, num_sampled_, num_to_sample, var_star);
  } else {
    // compute as Ks^T * (K\ Ks), the 2nd term of which has been precomputed
    // this is cheaper than computing V^T * V when K \ Ks is already available
    GeneralMatrixMatrixMultiply(points_to_sample_state->K_star.data(), 'T',
                                points_to_sample_state->K_inv_times_K_star.data(),
                                -1.0, 1.0, num_to_sample, num_sampled_, num_to_sample, var_star);
  }
}

/*!\rst
  **CORE IDEA**

  Similar to ComputeGradCholeskyVarianceOfPoints() below, except this function does not account for the cholesky decomposition.  That is,
  it produces derivatives wrt ``Xs_{d,p}`` (``points_to_sample``) of:

  ``Vars = Kss - (V^T * V) = Kss - Ks^T * K^-1 * Ks`` (see ComputeVarianceOfPoints)

  .. NOTE:: normally ``Xs_p`` would be the ``p``-th point of Xs (all dimensions); here ``Xs_{d,p}`` more explicitly
      refers to the ``d``-th spatial dimension of the ``p``-th point.

  This function only returns the derivative wrt a single choice of ``p``, as specified by ``diff_index``.

  Expanded index notation:

  ``Vars_{i,j} = Kss_{i,j} - Ks^T_{i,l} * K^-1_{l,k} * Ks_{k,j}``

  Recall ``Ks_{k,i} = cov(X_k, Xs_i) = cov(Xs_i, Xs_k)`` where ``Xs`` is ``points_to_sample`` and ``X`` is ``points_sampled``.
  (Note this is not equivalent to saying ``Ks = Ks^T``, although this would be true if ``|Xs| == |X|``.)
  As a result of this symmetry, ``\pderiv{Ks_{k,i}}{Xs_{d,i}} = \pderiv{Ks_{i,k}}{Xs_{d,i}}`` (that's ``d(cov(Xs_i, X_k))/d(Xs_i)``)

  We are being more strict with index labels than is standard to clearly specify tensor dimensions.  To be clear:
  1. ``i,j`` range over ``num_to_sample``
  2. ``l,k`` are the only non-free indices; they range over ``num_sampled``
  3. ``d,p`` describe the SPECIFIC point being differentiated against in ``Xs`` (``points_to_sample``): ``d`` over dimension, ``p``\* over ``num_to_sample``

  \*NOTE: ``p`` is *fixed*! Unlike all other indices, ``p`` refers to a *SPECIFIC* point in the range ``[0, ..., num_to_sample-1]``.
          Thus, ``\pderiv{Ks_{k,i}}{Xs_{d,i}}`` is a 3-tensor (``A_{d,k,i}``) (repeated ``i`` is not summation since they denote
          components of a derivative) while ``\pderiv{Ks_{i,l}}{Xs_{d,p}}`` is a 2-tensor (``A_{d,l}``) b/c only
          ``\pderiv{Ks_{i=p,l}}{Xs_{d,p}}`` is nonzero, and ``{d,l}`` are the only remaining free indices.

  Then differentiating against ``Xs_{d,p}`` (recall that this is a specific point b/c p is fixed):

  | ``\pderiv{Vars_{i,j}}{Xs_{d,p}} = \pderiv{K_ss{i,j}}{Xs_{d,p}} -``
  | ``(\pderiv{Ks_{i,l}}{Xs_{d,p}} * K^-1_{l,k} * Ks_{k,j}   +  K_s{i,l} * K^-1_{l,k} * \pderiv{Ks_{k,j}}{Xs_{d,p}})``

  Many of these terms are analytically known to be 0: ``\pderiv{Ks_{i,l}}{Xs_{d,p}} = 0`` when ``p != i`` (see NOTE above).
  A similar statement holds for the other gradient term.

  Observe that the second term in the parens, ``Ks_{i,l} * K^-1_{l,k} * \pderiv{Ks_{k,j}}{Xs_{d,p}}``, can be reordered
  to "look" like the first term.  We use three symmetries: ``K^-1{l,k} = K^-1{k,l}``, ``Ks_{i,l} = Ks_{l,i}``, and

  ``\pderiv{Ks_{k,j}}{Xs_{d,p}} = \pderiv{Ks_{j,k}}{Xs_{d,p}}``

  Then we can write:

  ``K_s{i,l} * K^-1_{l,k} * \pderiv{Ks_{k,j}}{Xs_{d,p}} = \pderiv{Ks_{j,k}}{Xs_{d,p}} * K^-1_{k,l} * K_s{l,i}``

  Now left and right terms have the same index ordering (i,j match; k,l are not free and thus immaterial)

  The final result, accounting for analytic zeros is given here for convenience::

    DVars_{d,i,j} \equiv \pderiv{Vars_{i,j}}{Xs_{d,p}} =``
      { \pderiv{K_ss{i,j}}{Xs_{d,p}} - 2*\pderiv{Ks_{i,l}}{Xs_{d,p}} * K^-1_{l,k} * Ks_{k,j}   :  WHEN p == i == j
      { \pderiv{K_ss{i,j}}{Xs_{d,p}} -   \pderiv{Ks_{i,l}}{Xs_{d,p}} * K^-1_{l,k} * Ks_{k,j}   :  WHEN p == i != j
      { \pderiv{K_ss{i,j}}{Xs_{d,p}} -   \pderiv{Ks_{j,k}}{Xs_{d,p}} * K^-1_{k,l} * K_s{l,i}   :  WHEN p == j != i
      {                                    0                                                   :  otherwise

  The first item has a factor of 2 b/c it gets a contribution from both parts of the sum since ``p == i`` and ``p == j``.
  The ordering ``DVars_{d,i,j}`` is significant: this is the ordering (d changes the fastest) in storage.

  **OPTIMIZATIONS**

  Implementing this formula naively results in a large amount of redundant computation, so we now describe the optimizations
  present in our implementation.

  The first thing to notice is that the result, ``\pderiv{Vars_{i,j}}{Xs_{d,p}}``, has a lot of 0s.  In particular, only the
  ``p``-th block row and ``p``-th block column have nonzero entries (blocks are size ``dim``, indexed ``d``).  Currently,
  we will not be taking advantage of this sparsity because the consumer of DVars, ComputeGradCholeskyVarianceOfPoints(),
  is not implemented with sparsity in mind.

  Similarly, the next thing to notice is that if we ignore the case ``p == i == j``, then we see that the expressions for
  ``p == i`` and ``p == j`` are actually identical (e.g., take the ``p == j`` case and exchange ``j = i`` and ``k = l``).

  So think of ``DVars`` as a block matrix; each block has dimension entries, and the blocks are indexed over
  ``i`` (rows), ``j`` (cols).  Then we see that the code is block-symmetric: ``DVars_{d,i,j} = Dvars_{d,j,i}``.
  So we can compute it by filling in the ``p``-th block column and then copy that data into the ``p``-th block row.

  Additionally, the derivative terms represent matrix-matrix products:
  ``C_{l,j} = K^-1_{l,k} * Ks_{k,j}`` (and ``K^-1_{k,l} * Ks_{l,i}``, which is just a change of index labels) is
  a matrix product.  We compute this using back-substitutions to avoid explicitly forming ``K^-1``.  ``C_{l,j}``
  is ``num_sampled`` X ``num_to_sample``.

  Then ``D_{d,i=p,j} = \pderiv{Ks_{i=p,l}}{Xs_{d,p}} * C_{l,j}`` is another matrix product (result size ``dim * num_to_sample``)
  (``i = p`` indicates that index ``i`` collapses out since this deriv term is zero if ``p != i``).
  Note that we store ``\pderiv{Ks_{i=p,l}}{Xs_{d,p}} = \pderiv{Ks_{l,i=p}}{Xs_{d,p}}`` as ``A_{d,l,i}``
  and grab the ``i = p``-th block.

  Again, only the ``p``-th point of ``points_to_sample`` is differentiated against; ``p`` specfied in ``diff_index``.
\endrst*/
void GaussianProcess::ComputeGradVarianceOfPointsPerPoint(StateType * points_to_sample_state,
                                                          int diff_index, double * restrict grad_var) const noexcept {
  const int num_to_sample = points_to_sample_state->num_to_sample;

  // we only visit a small subset of the entries in this matrix; need to ensure the others are zero'd
  std::fill(grad_var, grad_var + dim_*Square(num_to_sample), 0.0);

  // Compute: \pderiv{Ks_{i,l}}{Xs_{d,p}} * K^-1_{l,k} * Ks_{k,j} (the second term in DVvars, above).
  // Retrieve C_{l,j} = K^-1_{l,k} * Ks_{k,j}, from C stored in K_inv_times_K_star
  // Retrieve \pderiv{Ks_{l,i=p}}{Xs_{d,p}} from state struct (stored as A_{d,l,p}), use in matrix product
  // Result is computed as: A_{d,l,p} * C_{l,j}.  (Again, recall that p is fixed, so this output is over a matrix indexed {d,j}.)
  double * restrict grad_var_target_column = grad_var + diff_index*dim_*num_to_sample;
  GeneralMatrixMatrixMultiply(points_to_sample_state->grad_K_star.data() + diff_index*dim_*num_sampled_, 'N',
                              points_to_sample_state->K_inv_times_K_star.data(), 1.0, 0.0,
                              dim_, num_sampled_, num_to_sample, grad_var_target_column);

  // Fill the p-th block column of the output (p = diff_index); we will then copy this into the p-th block column.
  for (int j = 0; j < num_to_sample; ++j) {
    // Compute the leading term: \pderiv{K_ss{i=p,j}}{Xs_{d,p}}.
    covariance_ptr_->GradCovariance(points_to_sample_state->points_to_sample.data() + diff_index*dim_,
                                    points_to_sample_state->points_to_sample.data() + j*dim_,
                                    points_to_sample_state->grad_cov.data());
    // Flip the sign, add leading term in.
    if (j == diff_index) {  // Block diagonal term needs to be multiplied by 2.
      for (int m = 0; m < dim_; ++m) {
        grad_var_target_column[m] *= -2.0;
        grad_var_target_column[m] += points_to_sample_state->grad_cov[m];
      }
    } else {
      for (int m = 0; m < dim_; ++m) {
        grad_var_target_column[m] *= -1.0;
        grad_var_target_column[m] += points_to_sample_state->grad_cov[m];
      }
    }
    grad_var_target_column += dim_;
  }

  grad_var_target_column -= dim_*num_to_sample;
  // Pointer to the first element of the block row we're filling.
  double * restrict grad_var_target_row = grad_var + diff_index*dim_;
  // Fill in the diff_index-th block row by copying from the diff_index-th block column.
  for (int j = 0; j < num_to_sample; ++j) {
    // Skip the diagonal block (we'd just be copying it onto itself).
    if (j != diff_index) {
      // From function comments, the matrix is block-symmetric so we just copy directly.
      for (int m = 0; m < dim_; ++m) {
        grad_var_target_row[m] = grad_var_target_column[m];
      }
    }
    grad_var_target_column += dim_;
    grad_var_target_row += num_to_sample*dim_;
  }
}

/*!\rst
  This is just a thin wrapper that calls ComputeGradVarianceOfPointsPerPoint() in a loop ``num_derivatives`` times.

  See ComputeGradVarianceOfPointsPerPoint()'s function comments and implementation for more mathematical details
  on the derivation, algorithm, optimizations, etc.
\endrst*/
void GaussianProcess::ComputeGradVarianceOfPoints(StateType * points_to_sample_state,
                                                  double * restrict grad_var) const noexcept {
  int block_size = Square(points_to_sample_state->num_to_sample)*dim_;
  for (int k = 0; k < points_to_sample_state->num_derivatives; ++k) {
    ComputeGradVarianceOfPointsPerPoint(points_to_sample_state, k, grad_var);
    grad_var += block_size;
  }
}

/*!\rst
  Differentiates the cholesky factorization of the GP variance.

  | ``Vars = Kss - (V^T * V)``  (see ComputeVarianceOfPoints)
  | ``C * C^T = Vars``

  This function differentiates ``C`` wrt the ``p``-th point of ``points_to_sample``; ``p`` specfied in ``diff_index``

  Just as users of a lower triangular matrix ``L[i][j]`` should not access the upper triangle (``j > i``), users of
  the result of this function, ``grad_chol[d][i][j]``, should not access the upper *block* triangle with ``j > i``.

  See Smith 1995 for full details of computing gradients of the cholesky factorization
\endrst*/
void GaussianProcess::ComputeGradCholeskyVarianceOfPointsPerPoint(StateType * points_to_sample_state,
                                                                  int diff_index, double const * restrict chol_var,
                                                                  double * restrict grad_chol) const noexcept {
  ComputeGradVarianceOfPointsPerPoint(points_to_sample_state, diff_index, grad_chol);

  // TODO(GH-173): Try reorganizing Smith's algorithm to use an ordering analogous to the gaxpy
  // formulation of cholesky (currently it's organized like the outer-product version which results in
  // more memory accesses).

  const int num_to_sample = points_to_sample_state->num_to_sample;
  // input is upper block triangular, zero the lower block triangle
  for (int i = 0; i < num_to_sample; ++i) {
    int end_index = dim_*num_to_sample;
    // In GV_{mji}, each j > i specifies a lower diagonal block; each block has dim_ elements.
    // So we start on the (i+1)-th block and go to the end of this block column.
    for (int j = (i+1)*dim_; j < end_index; ++j) {
      grad_chol[j] = 0.0;
    }
    grad_chol += num_to_sample*dim_;
  }
  grad_chol -= num_to_sample*num_to_sample*dim_;

  // Loop annotations match those in ComputeCholeskyFactorL() to describe what each segment differentiates and how.
  // In the following comments, L_{ij} := chol_var[j*num_to_sample + i] is the cholesky factorization of the variance,
  // and GV_{mij} := grad_chol[j*num_to_sample*dim_ + i*dim_ + m] is the gradient of variance (input),
  // and GL_{mij} := grad_chol[j*num_to_sample*dim_ + i*dim_ + m] is the gradient of cholesky of variance (on exit)
  // Define macros specifying the data layout assumption on L_{ij} and GV_{mij}. The macro simplifies complex indexing
  // so that OL_CHOL_VAR(i, j) reads just like L_{ij}, for example.
#define OL_CHOL_VAR(i, j) chol_var[((j)*num_to_sample + (i))]
#define OL_GRAD_CHOL(m, i, j) grad_chol[((j)*num_to_sample*dim_ + (i)*dim_ + (m))]

  for (int k = 0; k < num_to_sample; ++k) {
    // L_kk := L_{kk}
    const double L_kk = OL_CHOL_VAR(k, k);

    if (likely(L_kk > kMinimumStdDev)) {
      // differentiates L_kk := L_{kk}
      // GL_{mkk} = 0.5 * GV_{mkk}/L_{kk}
      for (int m = 0; m < dim_; ++m) {
        OL_GRAD_CHOL(m, k, k) = 0.5*OL_GRAD_CHOL(m, k, k)/L_kk;
      }

      // differentiates L_{jk} = L_{jk}/L_{kk}
      // GL_{mkj} = (GV_{mkj} - L_{jk}*GV_{mkk})/L_{kk}
      for (int j = k+1; j < num_to_sample; ++j) {
        for (int m = 0; m < dim_; ++m) {
          OL_GRAD_CHOL(m, k, j) = (OL_GRAD_CHOL(m, k, j) - OL_CHOL_VAR(j, k)*OL_GRAD_CHOL(m, k, k))/L_kk;
        }
      }  // end for j: num_to_sample

      // differentiates L_{ij} = L_{ij} - L_{ik}*L_{jk}
      // GL_{mji} = GV_{mji} - GV_{mki}*L_{jk} - L_{ik}*GV_{mkj}
      for (int j = k+1; j < num_to_sample; ++j) {
        for (int i = j; i < num_to_sample; ++i) {
          for (int m = 0; m < dim_; ++m) {
            OL_GRAD_CHOL(m, j, i) = OL_GRAD_CHOL(m, j, i)
                - OL_GRAD_CHOL(m, k, i)*OL_CHOL_VAR(j, k) - OL_CHOL_VAR(i, k)*OL_GRAD_CHOL(m, k, j);
          }
        }  // end for i: num_to_sample
      }  // end for j: num_to_sample
    } else {
      OL_ERROR_PRINTF("Grad Cholesky failed; matrix singular. k=%d\n", k);
    }  // end if: L_kk is not "too small"
  }  // end for k: sie_of_to_sample
#undef OL_CHOL_VAR
#undef OL_GRAD_CHOL
}

/*!\rst
  This is just a thin wrapper that calls ComputeGradCholeskyVarianceOfPointsPerPoint() in a loop ``num_derivatives`` times.

  See ComputeGradCholeskyVarianceOfPointsPerPoint()'s function comments and implementation for more mathematical
  details on the algorithm.
\endrst*/
void GaussianProcess::ComputeGradCholeskyVarianceOfPoints(StateType * points_to_sample_state,
                                                          double const * restrict chol_var,
                                                          double * restrict grad_chol) const noexcept {
  int block_size = Square(points_to_sample_state->num_to_sample)*dim_;
  for (int k = 0; k < points_to_sample_state->num_derivatives; ++k) {
    ComputeGradCholeskyVarianceOfPointsPerPoint(points_to_sample_state, k, chol_var, grad_chol);
    grad_chol += block_size;
  }
}

void GaussianProcess::AddPointsToGP(double const * restrict new_points,
                                    double const * restrict new_points_value,
                                    double const * restrict new_points_noise_variance,
                                    int num_new_points) {
  // update sizes
  num_sampled_ += num_new_points;

  // update state variables
  points_sampled_.resize(num_sampled_*dim_);
  std::copy_backward(new_points, new_points + num_new_points*dim_, points_sampled_.end());

  points_sampled_value_.resize(num_sampled_);
  std::copy_backward(new_points_value, new_points_value + num_new_points, points_sampled_value_.end());

  noise_variance_.resize(num_sampled_);
  std::copy_backward(new_points_noise_variance, new_points_noise_variance + num_new_points, noise_variance_.end());

  // recompute derived quantities
  // TODO(GH-192): Insert the new covariance (and cholesky covariance) rows into the current matrix  (O(N^2))
  // instead of recomputing everything (O(N^3)).
  RecomputeDerivedVariables();
}

/*!\rst
  Samples function values from a GPP given a list of points.

  Samples by: ``function_value = gpp_mean + gpp_variance * w``, where ``w`` is a single draw from N(0,1).

  We only draw one point at a time (i.e., ``num_to_sample`` fixed at 1).  We want multiple draws from the same GPP;
  drawing many points per step would be akin to sampling multiple GPPs. Thus gpp_mean, gpp_variance, and w all have size 1.

  If the GPP does not receive any data, then on the first step, gpp_mean = 0 and gpp_variance is just the "covariance"
  of a single point. Then we iterate through the remaining points in points_sampled, generating gpp_mean, gpp_variance,
  and a sample function value.
\endrst*/
double GaussianProcess::SamplePointFromGP(double const * restrict point_to_sample,
                                          double noise_variance_this_point) noexcept {
  double gpp_variance;
  double gpp_mean;
  const int num_to_sample = 1;  // we will only draw 1 point at a time from the GP

  if (unlikely(num_sampled_ == 0)) {
    BuildCovarianceMatrix(*covariance_ptr_, point_to_sample, dim_, num_to_sample, &gpp_variance);
    return std::sqrt(gpp_variance) * normal_rng_() + std::sqrt(noise_variance_this_point)*normal_rng_();  // first draw has mean 0
  } else {
    int num_derivatives = 0;
    StateType points_to_sample_state(*this, point_to_sample, num_to_sample, num_derivatives);

    ComputeMeanOfPoints(points_to_sample_state, &gpp_mean);
    ComputeVarianceOfPoints(&points_to_sample_state, &gpp_variance);

    return gpp_mean + std::sqrt(gpp_variance) * normal_rng_() + std::sqrt(noise_variance_this_point)*normal_rng_();
  }
}

void GaussianProcess::SetExplicitSeed(EngineType::result_type seed) noexcept {
  normal_rng_.SetExplicitSeed(seed);
}

void GaussianProcess::SetRandomizedSeed(EngineType::result_type seed) noexcept {
  normal_rng_.SetRandomizedSeed(seed, 0);  // this is intended for single-threaded use only, so thread_id = 0
}

void GaussianProcess::ResetToMostRecentSeed() noexcept {
  normal_rng_.ResetToMostRecentSeed();
}

GaussianProcess * GaussianProcess::Clone() const {
  return new GaussianProcess(*this);
}

void PointsToSampleState::SetupState(const GaussianProcess& gaussian_process, double const * restrict points_to_sample_in,
                                     int num_to_sample_in, int num_derivatives_in) {
  // resize data depending on to sample points
  if (unlikely(num_to_sample != num_to_sample_in || num_derivatives != num_derivatives_in)) {
    // update sizes
    num_to_sample = num_to_sample_in;
    num_derivatives = num_derivatives_in;
    // resize vectors
    points_to_sample.resize(dim*num_to_sample);
    K_star.resize(num_to_sample*num_sampled);
    grad_K_star.resize(num_derivatives*num_sampled*dim);
    V.resize(num_to_sample*num_sampled);
    K_inv_times_K_star.resize(num_to_sample*num_sampled);
  }

  // resize data depending on sampled points
  if (unlikely(num_sampled != gaussian_process.num_sampled())) {
    num_sampled = gaussian_process.num_sampled();
    K_star.resize(num_to_sample*num_sampled);
    grad_K_star.resize(num_to_sample*num_sampled*dim);
    V.resize(num_to_sample*num_sampled);
    K_inv_times_K_star.resize(num_to_sample*num_sampled);
  }

  // set new points to sample
  std::copy(points_to_sample_in, points_to_sample_in + dim*num_to_sample, points_to_sample.begin());

  gaussian_process.FillPointsToSampleState(this);
}

PointsToSampleState::PointsToSampleState(const GaussianProcess& gaussian_process,
                                         double const * restrict points_to_sample_in,
                                         int num_to_sample_in, int num_derivatives_in)
    : dim(gaussian_process.dim()),
      num_sampled(gaussian_process.num_sampled()),
      num_to_sample(num_to_sample_in),
      num_derivatives(num_derivatives_in),
      points_to_sample(dim*num_to_sample),
      K_star(num_to_sample*num_sampled),
      grad_K_star(num_derivatives*num_sampled*dim),
      V(num_to_sample*num_sampled),
      K_inv_times_K_star(num_to_sample*num_sampled),
      grad_cov(dim) {
  SetupState(gaussian_process, points_to_sample_in, num_to_sample_in, num_derivatives_in);
}

PointsToSampleState::PointsToSampleState(PointsToSampleState&& OL_UNUSED(other)) = default;

ExpectedImprovementEvaluator::ExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in,
                                                           int num_mc_iterations, double best_so_far)
    : dim_(gaussian_process_in.dim()),
      num_mc_iterations_(num_mc_iterations),
      best_so_far_(best_so_far),
      gaussian_process_(&gaussian_process_in) {
}

/*!\rst
  Let ``Ls * Ls^T = Vars`` and ``w`` = vector of IID normal(0,1) variables
  Then:

  ``y = mus + Ls * w``  (Equation 4, from file docs)

  simulates drawing from our GP with mean mus and variance Vars.

  Then as given in the file docs, we compute the improvement:
  Then the improvement for this single sample is::

    I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5 from file docs)
        {          0               else

  This is implemented as ``max_{y} (best_known - y)``.  Notice that improvement takes the value 0 if it would be negative.

  Since we cannot compute ``min(y)`` directly, we do so via monte-carlo (MC) integration.  That is, we draw from the GP
  repeatedly, computing improvement during each iteration, and averaging the result.

  See Scott's PhD thesis, sec 6.2.

  .. Note:: comments here are copied to _compute_expected_improvement_monte_carlo() in python_version/expected_improvement.py
\endrst*/
double ExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.", ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }

  double aggregate = 0.0;
  for (int i = 0; i < num_mc_iterations_; ++i) {
    double improvement_this_step = 0.0;
    for (int j = 0; j < num_union; ++j) {
      ei_state->EI_this_step_from_var[j] = (*(ei_state->normal_rng))();  // EI_this_step now holds "normals"
    }

    // compute EI_this_step_from_far = cholesky * normals   as  EI = cholesky * EI
    // b/c normals currently held in EI_this_step_from_var
    TriangularMatrixVectorMultiply(ei_state->cholesky_to_sample_var.data(), 'N', num_union,
                                   ei_state->EI_this_step_from_var.data());
    for (int j = 0; j < num_union; ++j) {
      double EI_total = best_so_far_ - (ei_state->to_sample_mean[j] + ei_state->EI_this_step_from_var[j]);
      if (EI_total > improvement_this_step) {
        improvement_this_step = EI_total;
      }
    }

    if (improvement_this_step > 0.0) {
      aggregate += improvement_this_step;
    }
  }

  return aggregate/static_cast<double>(num_mc_iterations_);
}

/*!\rst
  Computes gradient of EI (see ExpectedImprovementEvaluator::ComputeGradExpectedImprovement) wrt points_to_sample (stored in
  ``union_of_points[0:num_to_sample]``).

  Mechanism is similar to the computation of EI, where points' contributions to the gradient are thrown out of their
  corresponding ``improvement <= 0.0``.

  Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
  That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
  ``ei_state->grad_mu``).  The interaction with ``ei_state->grad_chol_decomp`` is harder to know a priori (like with
  ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
  the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.

  .. Note:: comments here are copied to _compute_grad_expected_improvement_monte_carlo() in python_version/expected_improvement.py
\endrst*/
void ExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  const int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, ei_state->grad_mu.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException, "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.", ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }

  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state),
                                                         ei_state->cholesky_to_sample_var.data(),
                                                         ei_state->grad_chol_decomp.data());

  std::fill(ei_state->aggregate.begin(), ei_state->aggregate.end(), 0.0);
  double aggregate_EI = 0.0;
  for (int i = 0; i < num_mc_iterations_; ++i) {
    for (int j = 0; j < num_union; ++j) {
      ei_state->EI_this_step_from_var[j] = (*(ei_state->normal_rng))();  // EI_this_step now holds "normals"
      ei_state->normals[j] = ei_state->EI_this_step_from_var[j];  // orig value of normals needed if improvement_this_step > 0.0
    }

    // compute EI_this_step_from_far = cholesky * normals   as  EI = cholesky * EI
    // b/c normals currently held in EI_this_step_from_var
    TriangularMatrixVectorMultiply(ei_state->cholesky_to_sample_var.data(), 'N', num_union,
                                   ei_state->EI_this_step_from_var.data());

    double improvement_this_step = 0.0;
    int winner = num_union + 1;  // an out of-bounds initial value
    for (int j = 0; j < num_union; ++j) {
      double EI_total = best_so_far_ - (ei_state->to_sample_mean[j] + ei_state->EI_this_step_from_var[j]);
      if (EI_total > improvement_this_step) {
        improvement_this_step = EI_total;
        winner = j;
      }
    }

    if (improvement_this_step > 0.0) {
      // improvement > 0.0 implies winner will be valid; i.e., in 0:ei_state->num_to_sample
      aggregate_EI += improvement_this_step;

      // recall that grad_mu only stores \frac{d mu_i}{d Xs_i}, since \frac{d mu_j}{d Xs_i} = 0 for i != j.
      // hence the only relevant term from grad_mu is the one describing the gradient wrt winner-th point,
      // and this term only arises if the winner (for most improvement) index is less than num_to_sample
      if (winner < ei_state->num_to_sample) {
        for (int k = 0; k < dim_; ++k) {
          ei_state->aggregate[winner*dim_ + k] -= ei_state->grad_mu[winner*dim_ + k];
        }
      }

      // let L_{d,i,j,k} = grad_chol_decomp, d over dim_, i, j over num_union, k over num_to_sample
      // we want to compute: agg_dx_{d,k} = L_{d,i,j=winner,k} * normals_i
      // TODO(GH-92): Form this as one GeneralMatrixVectorMultiply() call by storing data as L_{d,i,k,j} if it's faster.
      double const * restrict grad_chol_decomp_winner_block = ei_state->grad_chol_decomp.data() + winner*dim_*(num_union);
      for (int k = 0; k < ei_state->num_to_sample; ++k) {
        GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N', ei_state->normals.data(), -1.0, 1.0,
                                    dim_, num_union, dim_, ei_state->aggregate.data() + k*dim_);
        grad_chol_decomp_winner_block += dim_*Square(num_union);
      }
    }  // end if: improvement_this_step > 0.0
  }  // end for i: num_mc_iterations_

  for (int k = 0; k < ei_state->num_to_sample*dim_; ++k) {
    grad_EI[k] = ei_state->aggregate[k]/static_cast<double>(num_mc_iterations_);
  }
}

void ExpectedImprovementState::SetCurrentPoint(const EvaluatorType& ei_evaluator,
                          double const * restrict points_to_sample) {
  // update points_to_sample in union_of_points
  std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

  // evaluate derived quantities for the GP
  points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), union_of_points.data(),
                                    num_union, num_derivatives);
}

ExpectedImprovementState::ExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                                   double const * restrict points_to_sample,
                                                   double const * restrict points_being_sampled,
                                                   int num_to_sample_in, int num_being_sampled_in,
                                                   bool configure_for_gradients, NormalRNGInterface * normal_rng_in)
    : dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      num_union(num_to_sample + num_being_sampled),
      union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
      points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
      normal_rng(normal_rng_in),
      to_sample_mean(num_union),
      grad_mu(dim*num_derivatives),
      cholesky_to_sample_var(Square(num_union)),
      grad_chol_decomp(dim*Square(num_union)*num_derivatives),
      EI_this_step_from_var(num_union),
      aggregate(dim*num_derivatives),
      normals(num_union) {
}

ExpectedImprovementState::ExpectedImprovementState(ExpectedImprovementState&& OL_UNUSED(other)) = default;

void ExpectedImprovementState::SetupState(const EvaluatorType& ei_evaluator,
                                          double const * restrict points_to_sample) {
  if (unlikely(dim != ei_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(ei_evaluator, points_to_sample);
}

OnePotentialSampleExpectedImprovementEvaluator::OnePotentialSampleExpectedImprovementEvaluator(
    const GaussianProcess& gaussian_process_in,
    double best_so_far)
    : dim_(gaussian_process_in.dim()),
      best_so_far_(best_so_far),
      normal_(0.0, 1.0),
      gaussian_process_(&gaussian_process_in) {
}

/*!\rst
  Uses analytic formulas to compute EI when ``num_to_sample = 1`` and ``num_being_sampled = 0`` (occurs only in 1,0-EI).
  In this case, the single-parameter (posterior) GP is just a Gaussian.  So the integral in EI (previously eval'd with MC)
  can be computed 'exactly' using high-accuracy routines for the pdf & cdf of a Gaussian random variable.

  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
double OnePotentialSampleExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double to_sample_mean;
  double to_sample_var;

  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, &to_sample_mean);
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), &to_sample_var);
  to_sample_var = std::sqrt(std::fmax(kMinimumVarianceEI, to_sample_var));

  double temp = best_so_far_ - to_sample_mean;
  double EI = temp*boost::math::cdf(normal_, temp/to_sample_var) + to_sample_var*boost::math::pdf(normal_, temp/to_sample_var);

  return std::fmax(0.0, EI);
}

/*!\rst
  Differentiates OnePotentialSampleExpectedImprovementEvaluator::ComputeExpectedImprovement wrt
  ``points_to_sample`` (which is just ONE point; i.e., 1,0-EI).
  Again, this uses analytic formulas in terms of the pdf & cdf of a Gaussian since the integral in EI (and grad EI)
  can be evaluated exactly for this low dimensional case.

  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
void OnePotentialSampleExpectedImprovementEvaluator::ComputeGradExpectedImprovement(
    StateType * ei_state,
    double * restrict exp_grad_EI) const {
  double to_sample_mean;
  double to_sample_var;

  double * restrict grad_mu = ei_state->grad_mu.data();
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, &to_sample_mean);
  gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, grad_mu);
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), &to_sample_var);
  to_sample_var = std::fmax(kMinimumVarianceGradEI, to_sample_var);
  double sigma = std::sqrt(to_sample_var);

  double * restrict grad_chol_decomp = ei_state->grad_chol_decomp.data();
  // there is only 1 point, so gradient wrt 0-th point
  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state), &sigma, grad_chol_decomp);

  double mu_diff = best_so_far_ - to_sample_mean;
  double C = mu_diff/sigma;
  double pdf_C = boost::math::pdf(normal_, C);
  double cdf_C = boost::math::cdf(normal_, C);

  for (int i = 0; i < dim_; ++i) {
    double d_C = (-sigma*grad_mu[i] - grad_chol_decomp[i]*mu_diff)/to_sample_var;
    double d_A = -grad_mu[i]*cdf_C + mu_diff*pdf_C*d_C;
    double d_B = grad_chol_decomp[i]*pdf_C + sigma*(-C)*pdf_C*d_C;

    exp_grad_EI[i] = d_A + d_B;
  }
}

void OnePotentialSampleExpectedImprovementState::SetCurrentPoint(const EvaluatorType& ei_evaluator,
                                                                    double const * restrict point_to_sample_in) {
  // update current point in union_of_points
  std::copy(point_to_sample_in, point_to_sample_in + dim, point_to_sample.data());

  // evaluate derived quantities
  points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), point_to_sample.data(),
                                    num_to_sample, num_derivatives);
}

OnePotentialSampleExpectedImprovementState::OnePotentialSampleExpectedImprovementState(
    const EvaluatorType& ei_evaluator,
    double const * restrict point_to_sample_in,
    bool configure_for_gradients)
    : dim(ei_evaluator.dim()),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      point_to_sample(point_to_sample_in, point_to_sample_in + dim),
      points_to_sample_state(*ei_evaluator.gaussian_process(), point_to_sample.data(), num_to_sample, num_derivatives),
      grad_mu(dim*num_derivatives),
      grad_chol_decomp(dim*num_derivatives) {
}

OnePotentialSampleExpectedImprovementState::OnePotentialSampleExpectedImprovementState(
    const EvaluatorType& ei_evaluator,
    double const * restrict points_to_sample,
    double const * restrict OL_UNUSED(points_being_sampled),
    int OL_UNUSED(num_to_sample_in),
    int OL_UNUSED(num_being_sampled_in),
    bool configure_for_gradients,
    NormalRNGInterface * OL_UNUSED(normal_rng_in))
    : OnePotentialSampleExpectedImprovementState(ei_evaluator, points_to_sample, configure_for_gradients) {
}

OnePotentialSampleExpectedImprovementState::OnePotentialSampleExpectedImprovementState(
    OnePotentialSampleExpectedImprovementState&& OL_UNUSED(other)) = default;

void OnePotentialSampleExpectedImprovementState::SetupState(const EvaluatorType& ei_evaluator,
                                                            double const * restrict point_to_sample_in) {
  if (unlikely(dim != ei_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
  }

  SetCurrentPoint(ei_evaluator, point_to_sample_in);
}

/*!\rst
  Routes the EI computation through MultistartOptimizer + NullOptimizer to perform EI function evaluations at the list of input
  points, using the appropriate EI evaluator (e.g., monte carlo vs analytic) depending on inputs.
\endrst*/
void EvaluateEIAtPointList(const GaussianProcess& gaussian_process, const ThreadSchedule& thread_schedule,
                           double const * restrict initial_guesses, double const * restrict points_being_sampled,
                           int num_multistarts, int num_to_sample, int num_being_sampled, double best_so_far,
                           int max_int_steps, bool * restrict found_flag, NormalRNG * normal_rng,
                           double * restrict function_values, double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  using DomainType = DummyDomain;
  DomainType dummy_domain;
  bool configure_for_gradients = false;
  if (num_to_sample == 1 && num_being_sampled == 0) {
    // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);

    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, thread_schedule.max_num_threads,
                                  configure_for_gradients, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, initial_guesses);

    NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<OnePotentialSampleExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, dummy_domain,
                                            thread_schedule, initial_guesses, num_multistarts,
                                            ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  } else {
    ExpectedImprovementEvaluator ei_evaluator(gaussian_process, max_int_steps, best_so_far);

    std::vector<typename ExpectedImprovementEvaluator::StateType> ei_state_vector;
    SetupExpectedImprovementState(ei_evaluator, initial_guesses, points_being_sampled, num_to_sample,
                                  num_being_sampled, thread_schedule.max_num_threads,
                                  configure_for_gradients, normal_rng, &ei_state_vector);

    // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
    OptimizationIOContainer io_container(ei_state_vector[0].GetProblemSize(), 0.0, initial_guesses);

    NullOptimizer<ExpectedImprovementEvaluator, DomainType> null_opt;
    typename NullOptimizer<ExpectedImprovementEvaluator, DomainType>::ParameterStruct null_parameters;
    MultistartOptimizer<NullOptimizer<ExpectedImprovementEvaluator, DomainType> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters, dummy_domain,
                                            thread_schedule, initial_guesses, num_multistarts,
                                            ei_state_vector.data(), function_values, &io_container);
    *found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
  }
}

/*!\rst
  This is a simple wrapper around ComputeOptimalPointsToSampleWithRandomStarts() and
  ComputeOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).

  TODO(GH-77): Instead of random search, we may want to fall back on the methods in
  ``gpp_heuristic_expected_improvement_optimization.hpp`` if gradient descent fails; esp for larger q
  (even ``q \approx 4``), latin hypercube search does a pretty terrible job.
  This is more for general q,p-EI as these two things are equivalent for 1,0-EI.
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
                                  NormalRNG * normal_rng, double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }

  std::vector<double> next_points_to_sample(gaussian_process.dim()*num_to_sample);

  bool found_flag_local = false;
  if (lhc_search_only == false) {
    ComputeOptimalPointsToSampleWithRandomStarts(gaussian_process, optimizer_parameters,
                                                 domain, thread_schedule, points_being_sampled,
                                                 num_to_sample, num_being_sampled,
                                                 best_so_far, max_int_steps,
                                                 &found_flag_local, uniform_generator, normal_rng,
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
      ComputeOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, domain,
                                                          thread_schedule_naive_search,
                                                          points_being_sampled,
                                                          num_lhc_samples, num_to_sample,
                                                          num_being_sampled, best_so_far,
                                                          max_int_steps,
                                                          &found_flag_local, uniform_generator,
                                                          normal_rng, next_points_to_sample.data());

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
template void ComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, int num_to_sample,
    int num_being_sampled, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
template void ComputeOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled, double best_so_far, int max_int_steps,
    bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);

}  // end namespace optimal_learning
