/*!
  \file gpp_test_utils.hpp
  \rst
  This file declares functions and classes that are useful for unit testing.  This includes some relative/absolute precision
  checks and a few mathematical utilities.

  The "big stuff" in this file is a class that defines the interface for a pingable function:

  ``PingableMatrixInputVectorOutputInterface``

  Then there is a ``PingDerivative()`` function that can conduct ping testing on any implementer of this interface.

  There's also a mock environment class that sets up quantities commonly needed by tests of GP functionality.
  Similarly, there is a mock data class that builds up "history" (points_sampled, points_sampled_value) by
  constructing a GP with random hyperparameters on a random domain.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_HPP_

#include <memory>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_geometry.hpp"
#include "gpp_random.hpp"

namespace boost {
template<class RealType>
class uniform_real;
}

namespace optimal_learning {

class CovarianceInterface;
struct UniformRandomGenerator;
class GaussianProcess;
class TensorProductDomain;
class SimplexIntersectTensorProductDomain;

/*!\rst
  Class to enable numerical and analytic differentiation of functions of the form:

  ``f_{k} = f(X_{d,i})``

  with derivatives taken wrt each member of ``X_{d,i}``,

  ``gradf_{k,d,i} = \frac{\partial f_k}{\partial X_{d,i}}``

  In the nomenclature used in the class:

  * ``d`` indexes over num_rows (set in GetInputSizes())
  * ``i`` indexes over num_cols (set in GetInputSizes())
  * ``k`` indexes over GetOutputSize()

  Typically ``d`` is the spatial_dimension of the problem.  So if ``i`` ranges over ``1 .. num_points``,
  then ``X_{d,i}`` is a matrix of num_points points each with dimension spatial_dim.
  And ``k`` refers to num_outputs.

  ``X_{d,i}`` can of course be any arbitrary matrix; ``d, i`` need not refer to spatial dimension and num_points.  But
  that is currently the most common use case.

  This class enables easy pinging of a multitude of ``f, X`` combinations.  Since it abstracts away indexing, it does not limit
  how implementations store/compute ``f()`` and its gradient.

  Generally, usage goes as follows:

  * Use GetInputSizes(), GetOutputSize(), and possibly GetGradientsSize() to inspect the dimensions of the problem
  * EvaluateAndStoreAnalyticGradient(): compute and internally store the gradient evaluated at a given input\*
  * GetAnalyticGradient: returns the value of the analytic gradient for a given output (``k``), wrt a given point ``(d,i)``
  * EvaluateFunction: returns all outputs of the function for a given input

  \* It is not necessary to fully evaluate the gradient here.  Instead, the input point can be stored and evaluation can
  happen on-the-fly in GetAnalyticGradient() if desired.

  So to ping a derivative, you can:

  ``f_p = EvaluateFunction(X + h)``, ``f_m = EvaluateFunction(X - h)``

  Compare:

  ``(f_p - f_m)/(2h)``

  to

  ``GetAnalyticGradient``

  See ``PingDerivative()`` docs for more details.
\endrst*/
class PingableMatrixInputVectorOutputInterface {
 public:
  /*!\rst
    Number of rows and columns of the input, ``X_{d,i}``, to ``f()``.

    For example, the input might be a ``N_d X N_i`` matrix, ``points_to_sample``, where ``N_d`` = spatial dimension (rows)
    and ``N_i`` = number of points (columns).

    \output
      :num_rows[1]: the number of rows of the input matrix ``X``
      :num_cols[1]: the number of columns of the input matrix ``X``
  \endrst*/
  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept OL_NONNULL_POINTERS = 0;

  /*!\rst
    Number of outputs of the function ``f_k = f(X_{d,i})``; i.e., ``length(f_k)``

    \return
      The number of entries in ``f_k`` aka number of outputs of ``f()``
  \endrst*/
  virtual int GetOutputSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Number of entries in the gradient of the output wrt each entry of the input.

    This should generally not be used unless you require direct access to the analytic gradient.

    \return
      MUST be ``num_rows*num_cols*GetOutputSize()`` invalid memory read/writes may occur
  \endrst*/
  virtual int GetGradientsSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Setup so that GetAnalyticGradient(row, column, output) will be able to return
    ``gradf[row][column][output]`` evaluated at ``X``, "input_matrix."

    Typically this will entail computing and storing the analytic gradient.  But the only thing that needs
    to be saved is the contents of input_matrix for later access.

    MUST BE CALLED before using GetAnalyticGradient!

    \param
      :input_matrix[num_rows][num_cols]: the input, ``X_{d,i}``
    \output
      :gradients[num_rows][num_cols][num_outputs]: filled with the gradient evaluated at
        input_matrix.  Ignored if nullptr.  IMPLEMENTATION NOT REQUIRED.
  \endrst*/
  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict input_matrix, double * restrict gradients) noexcept OL_NONNULL_POINTERS_LIST(2) = 0;

  /*!\rst
    The gradients are indexed by: ``dA[input_row][input_column][output_index]``, where row, column index
    the input matrix and output_index indexes the output.

    Returns the gradient computed/stored by EvaluateAndStoreAnalyticGradient().

    \param
      :row_index: row_index (``d``) of the input to be differentiated with respect to
      :column_index: column_index (``i``) of the input to be differentiated with respect to
      :output_index: (flat) index into the output
    \return
      The ``[row_index][column_index][output_index]'th`` entry of the analytic gradient evaluated at input_matrix (where
      input matrix was specified in EvaluateAndStoreAnalyticGradient()).
  \endrst*/
  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Evalutes ``f_k = f(X_{d,i})``.  ``X_{d,i}`` is the "input_matrix" and ``f_k`` is in "function_values."

    \param
      :input_matrix[num_rows][num_cols]: the matrix of inputs
    \output
      :function_values[num_outputs]: vector of outputs of ``f()``
  \endrst*/
  virtual void EvaluateFunction(double const * restrict input_matrix, double * restrict function_values) const noexcept OL_NONNULL_POINTERS = 0;

  OL_DISALLOW_COPY_AND_ASSIGN(PingableMatrixInputVectorOutputInterface);

 protected:
  PingableMatrixInputVectorOutputInterface() = default;

  virtual ~PingableMatrixInputVectorOutputInterface() = default;
};

/*!\rst
  Class to conveniently hold and generate random data that are commonly needed for testing functions in gpp_math.cpp.  In
  particular, this mock is used for testing GP mean, GP variance, and expected improvement (and their gradients).

  This class holds arrays: ``points_sampled``, ``points_sampled_value``, ``points_to_sample``, and ``points_being_sampled``
  which are sized according to the parameters specified in Initialize(), and filled with random numbers.

  TODO(GH-125): we currently generate the point sets by repeated calls to rand().  This is generally
  unwise since the distribution of points is not particularly random.  Additionally, our current covariance
  functions are all stationary, so we would rather generate a random base point ``x``, and then a random
  (direction, radius) pair so that ``y = x + direction*radius``. We better cover the different behavioral
  regimes of our code in this case, since it's the radius value that actually correlates to results.
\endrst*/
class MockExpectedImprovementEnvironment {
 public:
  using EngineType = UniformRandomGenerator::EngineType;

  //! default seed for repeatability in testing
  static constexpr EngineType::result_type kDefaultSeed = 314;
  //! minimum coordinate value
  static constexpr double range_min = -5.0;
  //! maximum coordinate value
  static constexpr double range_max = 5.0;

  /*!\rst
    Construct a MockExpectedImprovementEnvironment and set invalid values for all size parameters
    (so that Initialize must be called to do anything useful) and pre-allocate some space.
  \endrst*/
  MockExpectedImprovementEnvironment();

  /*!\rst
    (Re-)initializes the data data in this function: this includes space allocation and random number generation.

    If any of the size parameters are changed from their current values, space will be realloc'd.
    Then it re-draws another set of uniform random points (in [-5, 5]) for the member arrays
    points_sampled, points_sampled_value, points_to_sample, and points_being_sampled.

    \param
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_to_sample: number of points to be sampled in future experiments
      :num_being_sampled: number of points being sampled concurrently
      :num_sampled: number of already-sampled points
  \endrst*/
  void Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in, int num_sampled_in) {
    Initialize(dim_in, num_to_sample_in, num_being_sampled_in, num_sampled_in, &uniform_generator_);
  }

  void Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in, int num_sampled_in, UniformRandomGenerator * uniform_generator);

  //! spatial dimension (e.g., entries per point of points_sampled)
  int dim;
  //! number of points in points_sampled (history)
  int num_sampled;
  //! number of points to be sampled in future experiments (i.e., the q in q,p-EI)
  int num_to_sample;
  //! number of points currently being sampled (i.e., the p in q,p-EI)
  int num_being_sampled;

  double * points_sampled() {
    return points_sampled_.data();
  }

  double * points_sampled_value() {
    return points_sampled_value_.data();
  }

  double * points_to_sample() {
    return points_to_sample_.data();
  }

  double * points_being_sampled() {
    return points_being_sampled_.data();
  }

  OL_DISALLOW_COPY_AND_ASSIGN(MockExpectedImprovementEnvironment);

 private:
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! points to be sampled in experiments (i.e., the q in q,p-EI)
  std::vector<double> points_to_sample_;
  //! points being sampled in concurrent experiments (i.e., the p in q,p-EI)
  std::vector<double> points_being_sampled_;

  //! uniform random number generator for generating coordinates
  UniformRandomGenerator uniform_generator_;
  //! distribution over ``[min, max]`` that coordinate values lie in
  boost::uniform_real<double> uniform_double_;
};

/*!\rst
  Struct to generate data randomly from a GaussianProcess. This object contains the generated GaussianProcess
  as well as the inputs needed to generate it (e.g., hyperparameters, domain, etc).

  This struct is intended for convenience (so that test writers do not need to repeat these lines in every test
  that builds its input data from a GP). It has a fire-and-forget constructor that builds all fields
  randomly, but it also exposes all internal state/functions used by that ctor so that power users
  can customize their test scenarios further.

  Implementation: this object uses std::unique_ptr to hide complex object definitions in the corresponding cpp file
\endrst*/
template <typename DomainType>
struct MockGaussianProcessPriorData {
  /*!\rst
    Construct an empty MockGaussianProcessPriorData. Member int, double, and vectors are
    initialized appropriately. covariance_ptr is cloned from covariance.
    BUT domain_ptr and gaussian_process_ptr ARE NOT initialized. Use this class's member
    functions to properly initialize these more complex data.

    \param
      :covariance: the CovarianceInterface object encoding assumptions about the GP's behavior on our data
      :noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, i-th entry will be associated with the i-th point generated by the GP
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_sampled: number of already-sampled points (that we want the GP to hold)
  \endrst*/
  MockGaussianProcessPriorData(const CovarianceInterface& covariance, const std::vector<double>& noise_variance_in, int dim_in, int num_sampled_in);

  /*!\rst
    Completely constructs a MockGaussianProcessPriorData, initializing all fields.
    Builds a GP based on a randomly generated domain and hyperparameters.

    \param
      :covariance: the CovarianceInterface object encoding assumptions about the GP's behavior on our data
      :noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, i-th entry will be associated with the i-th point generated by the GP
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_sampled: number of already-sampled points (that we want the GP to hold)
      :uniform_double_domain_lower: ``[min, max]`` range from which to draw domain lower bounds
      :uniform_double_domain_upper: ``[min, max]`` range from which to draw domain upper bounds
      :uniform_double_hyperparameters: ``[min, max]`` range from which to draw hyperparameters
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
  \endrst*/
  MockGaussianProcessPriorData(const CovarianceInterface& covariance, const std::vector<double>& noise_variance_in, int dim_in, int num_sampled_in, const boost::uniform_real<double>& uniform_double_domain_lower, const boost::uniform_real<double>& uniform_double_domain_upper, const boost::uniform_real<double>& uniform_double_hyperparameters, UniformRandomGenerator * uniform_generator);

  /*!\rst
    Prevent inline destructor: the dtor of std::unique_ptr<T> needs access to T's dtor (b/c
    unique_ptr's dtor basically calls delete on T*). But we want to forward-declare all of our
    T objects, so the dtor must be defined in the cpp file where those defintions are visible.
  \endrst*/
  ~MockGaussianProcessPriorData();

  /*!\rst
    Sets hyperparameters of covariance with random draws from the specified interval.
    Modifies: hyperparameters, covariance_ptr

    \param
      :uniform_double_hyperparameters: ``[min, max]`` range from which to draw hyperparameters
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
  \endrst*/
  void InitializeHyperparameters(const boost::uniform_real<double>& uniform_double_hyperparameters, UniformRandomGenerator * uniform_generator);

  /*!\rst
    Sets the domain from which the GP's historical data will be generated. For each dimension,
    we draw ``[min, max]`` bounds (domain_bounds); then we construct a domain object.
    Modifies: domain_bounds, domain_ptr

    \param
      :uniform_double_domain_lower: ``[min, max]`` range from which to draw domain lower bounds
      :uniform_double_domain_upper: ``[min, max]`` range from which to draw domain upper bounds
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
  \endrst*/
  void InitializeDomain(const boost::uniform_real<double>& uniform_double_domain_lower, const boost::uniform_real<double>& uniform_double_domain_upper, UniformRandomGenerator * uniform_generator);

  /*!\rst
    Builds a GaussianProcess with num_sampled points (drawn randomly from the domain) whose
    values are drawn randomly from the GP (sampled one at a time and added to the prior).
    Users MUST call InitializeHyperparameters() and InitializeDomain() (or otherwise initialize
    hyperparameters and domain_ptr) before calling this function.
    Modifies: covariance_ptr, best_so_far, gaussian_procss_ptr

    \param
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
  \endrst*/
  void InitializeGaussianProcess(UniformRandomGenerator * uniform_generator);

  //! spatial dimension (e.g., entries per point of points_sampled)
  int dim;
  //! number of points in points_sampled (history)
  int num_sampled;

  //! set of ``[min, max]`` bounds for the coordinates of each dimension
  std::vector<ClosedInterval> domain_bounds;
  //! the domain the GP will live in
  std::unique_ptr<DomainType> domain_ptr;

  //! covariance class (for computing covariance and its gradients); encodes assumptions about GP's behavior on data
  std::unique_ptr<CovarianceInterface> covariance_ptr;
  //! hyperparameters of the covariance function
  std::vector<double> hyperparameters;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance;
  //! lowest function value seen so far
  double best_so_far;
  //! the GP constructed by this object, using points sampled from domain and the stored covariance
  std::unique_ptr<GaussianProcess> gaussian_process_ptr;
};

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template struct MockGaussianProcessPriorData<TensorProductDomain>;
extern template struct MockGaussianProcessPriorData<SimplexIntersectTensorProductDomain>;

/*!\rst
  Checks if ``|value - truth| == 0``

  \param
    :value: number to be tested
    :truth: the exact/desired result
  \return
    true if value, truth are equal.
\endrst*/
bool CheckIntEquals(int64_t value, int64_t truth) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Computes ``||b - A*x||_2``

  The quantity ``b - A*x`` is called the "residual."  This is meaningful when x is
  the solution of the linear system ``A*x = b``.

  Coupled with knowledge of the underlying algorithm, having a small residual
  norm is a useful measure of method correctness.  See implementation documentation
  for more details.

  This norm is what is minimzied in least squares problems.  However here we
  are not working with least squares solutions and require that ``A`` is square.

  \param
    :A[size][size]: the linear system
    :x[size]: the solution vector
    :b[size]: the RHS vector
    :size: the dimension of the problem
  \return
    the 2-norm of b-A*x
\endrst*/
double ResidualNorm(double const * restrict A, double const * restrict x, double const * restrict b, int size) noexcept OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks if ``|value - truth| <= tolerance`` (error)

  \param
    :value: number to be tested
    :truth: the exact/desired result
    :tolerance: permissible difference
  \return
    true if value, truth differ by no more than tolerance.
\endrst*/
bool CheckDoubleWithin(double value, double truth, double tolerance) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks if ``|value - truth| / |truth| <= tolerance`` (relative error)

  If ``truth = 0.0``, ``CheckDoubleWithin()`` is performed.

  \param
    :value: number to be tested
    :truth: the exact/desired result
    :tolerance: permissible relative difference
  \return
    true if value, truth differ relatively by no more than tolerance.
\endrst*/
bool CheckDoubleWithinRelative(double value, double truth, double tolerance) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks if ``|value - truth| / |truth| <= tolerance`` (relative error)

  If ``truth < threshold``, ``CheckDoubleWithin()`` is performed.

  \param
    :value: number to be tested
    :truth: the exact/desired result
    :tolerance: permissible relative difference
    :threshold: ``tolerance = |value - truth|`` if ``|truth| < threshold``, this is to control unexpected large or
      undefined relative diff when truth is "too small" (0 for example)
  \return
    true if value, truth differ relatively by no more than tolerance.
\endrst*/
bool CheckDoubleWithinRelativeWithThreshold(double value, double truth, double tolerance, double threshold) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks that ``||A - B||_F <= tolerance``

  .. Note: the user may want to scale this norm by \sqrt(size) because ``||I||_F = \sqrt(size)``,
    and we may desire that the norm of the idenity be 1.

  \param
    :matrix1[size_m][size_n]: matrix A
    :matrix2[size_m][size_n]: matrix B
    :size_m: rows of A, B
    :size_n: columns of A, B
    :tolerance: largest permissible norm of the difference A - B
  \return
    true if A - B are "close"
\endrst*/
bool CheckMatrixNormWithin(double const * restrict matrix1, double const * restrict matrix2, int size_m, int size_n, double tolerance) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Check whether the distance between every pair of points is larger than tolerance.

  \param
    :point_list[dim][num_points]: list of points to check
    :dim: number of coordinates in each point
    :num_points: number of points
    :tolerance: the minimum allowed distance between points
  \return
    the number of pairs of points whose inter-point distance is less than tolerance
\endrst*/
int CheckPointsAreDistinct(double const * restrict point_list, int num_points, int dim, double tolerance) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks the correctness of analytic gradient calculations using finite differences.

  Since the exact level of error is virtually impossible to compute precisely, we
  use finite differences at two different h values (see implementation docs for
  details) and check the convergence rate.

  Includes logic to skip tests or run at relaxed tolerances when poor
  conditioning or loss precision is detected.

  In general, this function is meant to be used to test analytic gradient computations on a large
  number of random points. The logic to skip tests or relax tolerances is designed to decrease the
  false positive rate to 0. Some true positives are lost in the process. So we obtain "reasonable
  certainty" by testing a large number of points.

  If you are implementing/testing new gradients code, please read the function comments for this as well.
  This is an "expert tool" and not necessarily the most user-friendly at that.

  This function produces the most useful debugging output when in PingableMatrixInputVectorOutputInterface,

  ``num_rows`` = spatial dimension (``d``)

  ``num_cols`` = num_points (``i``)

  ``GetOutputSize() = num_outputs`` (``k``)

  for functions ``f_k = f(X_{d,i})``

  .. WARNING:: this function generates ~10 lines of output (to stdout) PER FAILURE. If your implementation
    is incorrect, expect a large number of lines printed to stdout.

  \param
    :function_and_derivative_evaluator: an object that inherits from
      PingableMatrixInputVectorOutputInterface. This object must define all
      virtual functions in that interface.
    :points[num_cols][num_rows]: num_cols points, each with dimension
      num_rows. Assumed that the coordinate-wise magnitues are "around 1.0": say [1.0e-3, 1.0e1].
    :epsilon[2]: ``array[h1, h2]`` of step sizes to use for finite differencing.
      These should not be too small/too large; 5.0e-3, 1.0e-3 are suggested
      starting values.
      Note that the more ill-conditioned computing f_k is, the larger the tolerances
      will need to be.  For example, "complex" functions (e.g., many math ops) may
      be ill-conditioned.
    :rate_tolerance_fine: desired amount of deviation from the exact rate
    :rate_tolerance_relaxed: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed
      Suggest values around 1.0e-15 to 1.0e-18 (around machine preciscion).
  \return
    The number of gradient entries that failed pinging.  Expected to be 0.
\endrst*/
int PingDerivative(const PingableMatrixInputVectorOutputInterface& function_and_derivative_evaluator, double const * restrict points, double epsilon[2], double rate_tolerance_fine, double rate_tolerance_relaxed, double input_output_ratio) noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

/*!\rst
  Expand each ClosedInterval by scale_factor (outward from their midpoints).

  This is handy for testing the constrained EI optimizers. That is, we want a domain which
  contains the true optima (b/c it is easier to know that things worked if the gradient is 0)
  but not a domain that is so large that constraint satisfaction is irrelevant. Hence we
  scale the side-lengths by some "reasonable" amount.

  \param
    :scale_factor: relative amount by which to expand each ClosedInterval
    :domain_bounds[1]: input list of valid ClosedInterval
  \output
    :domain_bounds[1]: input ClosedIntervals expanded by scale_factor
\endrst*/
void ExpandDomainBounds(double scale_factor, std::vector<ClosedInterval> * domain_bounds);

/*!\rst
  Generates random hyperparameters and sets a covariance object's hyperparameters to the new values.

  Let ``n_hyper = covariance->GetNumberOfHyperparameters()``.

  \param
    :uniform_double_hyperparameter: an uniform range, [min, max], from which to draw the hyperparameters
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :hyperparameters[1]: vector initialized to hold n_hyper items
    :covariance[1]: appropriately initialized covariance object that can accept n_hyper hyperparameters
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to hyperparameters->size() random draws
    :hyperparameters[1]: vector set to the newly generated hyperparameters
    :covariance[1]: covariance object with its hyperparameters set to "hyperparameters"
\endrst*/
void FillRandomCovarianceHyperparameters(const boost::uniform_real<double>& uniform_double_hyperparameter, UniformRandomGenerator * uniform_generator, std::vector<double> * hyperparameters, CovarianceInterface * covariance);

/*!\rst
  Generates a random list of ``domain_bounds->size()`` ``[min_i, max_i]`` pairs such that:

  ``min_i \in [uniform_double_lower_bound.a(), uniform_double_lower_bound.b()]``

  ``max_i \in [uniform_double_upper_bound.a(), uniform_double_upper_bound.b()]``

  \param
    :uniform_double_lower_bound: an uniform range, ``[min, max]``, from which to draw the domain lower bounds, ``min_i``
    :uniform_double_upper_bound: an uniform range, ``[min, max]``, from which to draw the domain upper bounds, ``max_i``
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :domain_bounds[1]: vector of ClosedInterval objects to initialize
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to ``2*domain_bounds->size()`` random draws
    :domain_bounds[1]: vector of ClosedInterval objects with their min, max members initialized as described
\endrst*/
void FillRandomDomainBounds(const boost::uniform_real<double>& uniform_double_lower_bound, const boost::uniform_real<double>& uniform_double_upper_bound, UniformRandomGenerator * uniform_generator, std::vector<ClosedInterval> * domain_bounds);

/*!\rst
  Utility to draw num_to_sample points from a GaussianProcess and add those values to the prior.

  \param
    :points_to_sample[dim][num_to_sample]: points at which to draw/sample from the GP
    :noise_variance[num_to_sample]: the ``\sigma_n^2`` (noise variance) associated w/the new observations, ``points_sampled_value``
    :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :num_to_sample: number of points add to the GP
    :gaussian_process[1]: a properly initialized GaussianProcess
  \output
    :points_to_sample_value[num_to_sample]: values of the newly sampled points
    :gaussian_process[1]: the input GP with num_sampled additional points added to it; its internal state is modified to match
                          the new data; and its internal PRNG state is modified
\endrst*/
void FillRandomGaussianProcess(double const * restrict points_to_sample, double const * restrict noise_variance, int dim, int num_to_sample, double * restrict points_to_sample_value, GaussianProcess * gaussian_process);

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_HPP_
