/*!
  \file gpp_optimization.hpp
  \rst
  Table of Contents:

  1. FILE OVERVIEW
  2. OPTIMIZATION OF OBJECTIVE FUNCTIONS

     a. GRADIENT DESCENT

        i. OVERVIEW
        ii. IMPLEMENTATION DETAILS

     b. NEWTON'S METHOD

        i. OVERVIEW
        ii. IMPLEMENTATION DETAILS

     c. MULTISTART OPTIMIZATION

  3. CODE HIERARCHY / CALL-TREE

     a. REQUIREMENTS OF TEMPLATE (CLASS) PARAMETERS
     b. CODE HIERARCHY / CALL-TREE FOR THIS FILE

        i. OPTIMIZER CLASS TEMPLATE
        ii. OPTIMIZER CLASSES IN THIS FILE
        iii. MULTISTART OPTIMIZATION

  .. Note:: comments in this header are copied in the module docstring in python/python_version/optimization.py
    and in the module docstring in python/cpp_wrappers/optimization.py.

  Read the "OVERVIEW" sections for header-style comments that describe the file contents at a high level.
  Read the "IMPLEMENTATION" comments for cpp-style comments that talk more about the specifics.  Both types
  are included together here since this file contains template class declarations and template function definitions.
  For further implementation details, see comment blocks before each individual class/function.

  **1. FILE OVERVIEW**

  First, the functions in this file are all MAXIMIZERS.  We also use the term "optima," and unless we specifically
  state otherwise, "optima" and "optimization" refer to "maxima" and "maximization," respectively.  (Note that
  minimizing ``g(x)`` is equivalent to maximizing ``f(x) = -1 * g(x)``.)

  This file contains templates for some common optimization techniques: gradient descent (GD) and Newton's method.
  We provide constrained implementations (constraint via heuristics like restricting updates to 50% of the distance
  to the nearest wall) of these optimizers.  For unconstrained, just set the domain to be huge: ``[-DBL_MAX, DBL_MAX]``.

  We provide \*Optimizer template classes (e.g., NewtonOptimizer) as main endpoints for doing local optimization
  (i.e., run the optimization method from a single initial guess).  We also provide a MultistartOptimizer class
  for global optimization (i.e., start optimizers from each of a set of initial guesses).  These are all discussed
  further below.

  All of the optimizers in this file are templated.  The template parameter is an Evaluator type which must have
  a type alias, Evaluator::StateType.  So we work with (Evaluator, State) "tuples."  The Evaluator encompasses data
  and functions needed to evaluate the objective function, its gradient, and/or its hessian.  The State covers
  the mutable state of the evaluation process.  So an Evaluator for ``f(x) = x^T * A * x / ||x||_2`` might contain the code for
  matrix-vector and vector-vector multiplication and the data for the matrix ``A``.  f's State would contain just the
  vector ``x``.  Then ``f::ComputeObjectiveFunction(state)`` would have everything it needs to compute ``f(x)``.

  In this way, we can make the local and global optimizers completely agonistic to the function they are optimizing.

  This Evaluator/State "idiom" is decsribed more thoroughly in item 5) in the header comments for gpp_common.hpp.  See
  section 3a) of this header (below) for details on precisely what functions/interface a (Evaluator, State) tuple are
  required to provide in order to be used with the optimizers in this file.

  **2. OPTIMIZATION OF OBJECTIVE FUNCTIONS**

  **2a. GRADIENT DESCENT (GD)**

  **2a, i. OVERVIEW**

  We use first derivative information to walk the path of steepest ascent, hopefully toward a (local) maxima of the
  chosen log likelihood measure.  This is implemented in: GradientDescentOptimization().
  This method ensures that the result lies within a specified domain.

  We additionally restart gradient-descent in practice; i.e., we repeatedly take the output of a GD run and start a
  new GD run from that point.  This lives in: GradientDescentOptimizer::Optimize().

  Even with restarts, gradient descent (GD) cannot start "too far" from the solution and still
  successfully find it.  Thus users should typically start it from multiple initial guesses and take the best one
  (see gpp_math and gpp_model_selection for examples).  The MultistartOptimizer template class in this file
  provides generic multistart functionality.

  Gradient descent is implemented in: GradientDescentOptimizer::Optimize() (which calls GradientDescentOptimization())

  **2a, ii. IMPLEMENTATION DETAILS**

  GD's update is: ``\theta_{i+1} = \theta_i + \gamma * \nabla f(\theta_i)``
  where ``\gamma`` controls the step-size and is chosen heuristically, often varying by problem.

  The previous update leads to unconstrained optimization.  To ensure that our results always stay within the
  specified domain, we additionally limit updates if they would move us outside the domain.  For example,
  we could imagine only moving half the distance to the nearest boundary.

  With gradient descent (GD), it is hard to know what step sizes to take.  Unfortunately, far enough away from an
  optima, the objective could increase (but very slowly).  If gradient descent takes too large of a step in a
  bad direction, it can easily "get lost."  At the same time, taking very small steps leads to slow performance.
  To help, we take the standard approach of scaling down step size with iteration number. We also allow the user
  to specify a maximum relative change to limit the aggressiveness of GD steps.  Finally, we wrap GD in a restart
  loop, where we fire off another GD run from the current location unless convergence was reached.

  **2b. NEWTON'S METHOD**

  **2b, i. OVERVIEW**

  Newton's Method (for optimization) uses second derivative information in addition to the first derivatives used by
  gradient descent (GD). In higher dimensions, first derivatives => gradients and second derivatives => Hessian matrix.
  At each iteration, gradient descent computes the derivative and blindly takes a step (of some
  heuristically determined size) in that direction.  Care must be taken in the step size choice to balance robustness
  and speed while ensuring that convergence is possible.  By using second derivative (the Hessian matrix in higher
  dimensions), which is interpretable as information about local curvature, Newton makes better\* choices about
  step size and direction to ensure rapid\*\* convergence.

  \*, \*\* See "IMPLEMENTATION DETAILS" comments section for details.

  Recall that Newton indiscriminately finds solutions where ``f'(x) = 0``; the eigenvalues of the Hessian classify these
  ``x`` as optima, saddle points, or indeterminate. We multistart Newton (e.g., gpp_model_selection)
  but just take the best objective value without classifying solutions.
  The MultistartOptimizer template class in this file provides generic multistart functionality.

  Newton is implemented here: NewtonOptimizer::Optimize() (which calls NewtonOptimization())

  **2b, ii. IMPLEMENTATION DETAILS**

  Let's address the footnotes from the previous section (Section 2b, i paragraph 1):

  \* Within its region of attraction, Newton's steps are optimal (when we have only second derivative information).  Outside
  of this region, Newton can make very poor decisions and diverge.  In general, Newton is more sensitive to its initial
  conditions than gradient descent, but it has the potential to be much, much faster.

  \*\* By quadratic convergence, we mean that once Newton is near enough to the solution, the log of the error will roughly
  halve each iteration.  Numerically, we would see the "number of digits" double each iteration.  Again, this only happens
  once Newton is "close enough."

  Newton's Method is a root-finding technique at its base.  To find a root of ``g(x)``, Newton requires an
  initial guess, ``x_0``, and the ability to compute ``g(x)`` and ``g'(x)``.  Then the idea is that you compute
  root of the line tangent to ``g(x_0)``; call this ``x_1``.  And repeat.  But the core idea is to make repeated
  linear approximations to ``g(x)`` and proceed in a fixed-point like fashion.

  As an optimization method, we are looking for roots of the gradient, ``f'(x_{opt}) = 0``.  So we require an initial guess
  ``x_0`` and the ability to evaluate ``f'(x)`` and ``f''(x)`` (in higher dimensions, the gradient and Hessian of f).  Thus Newton
  makes repeated linear approximations to ``f'(x)`` or equivalently, it locally approximates ``f(x)`` with a *quadratic* function,
  continuing iteration from the optima of that quadratic.
  In particular, Newton would solve the optimization problem of a quadratic program in one iteration.

  Mathematically, the update formulas for gradient descent (GD) and Newton are:
  GD:     ``\theta_{i+1} = \theta_i +     \gamma       * \nabla f(\theta_i)``
  Newton: ``\theta_{i+1} = \theta_i - H_f^-1(\theta_i) * \nabla f(\theta_i)``
  Note: the sign of the udpate is flipped because H is *negative* definite near a maxima.
  These update schemes are similar.  In GD, ``\gamma`` is chosen heuristically.  There are many ways to proceed but only
  so much that can be done with just gradient information; moreover the standard algorithm always proceeds in the direction
  of the gradient.  Newton takes a much more general appraoch.  Instead of a scalar ``\gamma``, the Newton update applies
  ``H^-1`` to the gradient, changing both the direction and magnitude of the step.

  Unfortunately, Newton indiscriminately finds solutions where ``f'(x) = 0``.  This is not necesarily an optima!  In one dimension,
  we can have ``f'(x) = 0`` and ``f''(x) = 0``, in which case the solution need not be an optima (e.g., ``y = x^3`` at ``x = 0``).
  In higher dimensions, a saddle point can also result (e.g., ``z = x^2 - y^2`` at ``x,y = 0``).  More generally, we have an
  optima if the Hessian is strictly negative or positive definite; a saddle if the Hessian has both positive and negative
  eigenvalues, and an indeterminate case if the Hessian is singular.

  **2c. MULTISTART OPTIMIZATION**

  Above, we mentioned that gradient descent (GD), Newton, etc. have a difficult time converging if they are started "too far"
  from an optima.  Even if convergence occurs, it will typically be very slow unless the problem is simple.  Worse,
  in a problem with multiple optima, the methods may converge to the wrong one!

  Multistarting the optimizers is one way of mitigating\* this issue.  Multistart involves starting a run of the
  specified optimizer (e.g., Newton) from each of a set of initial guesses.  Then the best result is reported as
  the result of the whole procedure.  By trying a large number of initial guesses, we potentially reduce the need
  for good guesses; i.e., hopefully at least one guess will be "near enough" to the global optimum.  This
  functionality is provided in MultistartOptimizer::MultistartOptimize(...).

  \* As noted below in the MultistartOptimizer::MultistartOptimize() function docs, mitigate is intentional here.
  Multistarting is NOT GUARANTEED to find global optima.  But it can increase the chances of success.

  Currently we let the user specify the initial guesses.  In practice, this typically means a random sampling of points.
  We do not (yet) make any effort to say sample more heavily from regions where "more stuff is happening" or any
  other heuristics.

  TODO(GH-165): Improve multistart heuristics.

  Finally, MultistartOptimizer::MultistartOptimize() is also used to provide 'dumb' search functionality (optimization
  by just evaluating the objective at numerous points).  For sufficiently complex problems, gradient descent, Newton, etc.
  can have exceptionally poor convergence characteristics or run too slowly.  In cases where these more advanced techniques
  fail, we commonly fall back to 'dumb' search.

  **3. CODE HIERARCHY / CALL-TREE**

  **3a. REQUIREMENTS OF TEMPLATE (CLASS) PARAMETERS**

  As mentioned in the overview, the functions and classes in this file are all templated on (Evaluator, State) tuples.
  They also template on Domain types.  In particular, the optimization functions and classes have the following form:
  template <typename ObjectiveFunctionEvaluator, typename Domain> optimization_function(...);
  template <typename ObjectiveFunctionEvaluator, typename Domain> OptimizationClass { ... };
  State (as indicated in gpp_common.hpp) is obtained through: typename ObjectiveFunctionEvaluator::State.

  Domain objects are explained in gpp_domain.hpp; see there for examples as well.  This file directly requires
  a Domain object to supply:
  void LimitUpdate(double max_relative_change, double const * restrict current_point, double * restrict update_vector);
  and with debugging on,
  bool CheckPointInside(double const * restrict point);

  Now let's talk about (Evaluate, State) template parameters and the optimzation classes in this file.  In ADDITION
  to the requirements/guidelines laid out in gpp_common.hpp, an (Evaluate, State) tuple MUST provide the following
  interface to be used with the optimizers in this file:

  Evaluator::

    // these functions all evaluate f() and its derivatives at state.GetCurrentPoint()
    // derivatives are computed against the space whose dimension is state.GetProblemSize()
    double ComputeObjectiveFunction(State * state);  // compute f(current_point)
    void ComputeGradObjectiveFunction(State * state, double * grad_objective);  // compute f'(current_point)
    void ComputeHessianObjectiveFunction(State * state, double * hessian_objective);  // compute f''(current_point)

  State::

    int GetProblemSize();  // how many dimensions to optimize
    void GetCurrentPoint(double * point);  // get current point at which Evalutor is computing results
    void SetCurrentPoint(double const * point);  // set current point at which Evalutor is computing results

  gpp_math.hpp and gpp_model_selection.hpp have (Evaluator, State) examples that implement
  the above interface:

  * gpp_math.hpp:

    * (ExpectedImprovement, ExpectedImprovementState)
    * (OnePotentialSampleExpectedImprovement, OnePotentialSampleExpectedImprovementState)

  * gpp_model_selection.hpp:

    * (LogMarginalLikelihoodEvaluator, LogMarginalLikelihoodState)
    * (LeaveOneOutLogLikelihoodEvaluator, LeaveOneOutLogLikelihoodState)

  gpp_mock_optimization_objective_functions.hpp lays out a pure abstract Evaluator class (with State) that is optimizable;
  examine tests that #include that file for more examples.

  .. Note:: not all (Evaluator, State) tuples (e.g., GaussianProcess) make sense for optimization; these classes
    do not provide the above interface.

  **3b. CODE HIERARCHY / CALL-TREE FOR THIS FILE**

  First, we will describe the interface for Optimizer classes.  Then we will go over individual classes.
  Finally we will touch on the MultistartOptimizer template class.

  **3b, i. OPTIMIZER CLASS TEMPLATE**

  As mentioned above, this file provides various Optimizer classes, e.g., NewtonOptimizer.  Here we'll go over high
  level details and then go through each specific example.

  TODO(GH-174): Include objective, param struct, domain, etc. as Optimizer class members (copies or references).

  In general, the Optimizer classes are the primary endpoint for local optimization; i.e., you have a good initial guess
  and you are confident that the optima is nearby.  In current use cases, this is uncommon.  But when it is true, multistart
  will waste a lot of compute time to arrive at the same result.
  For global optimization problems, use the MultistartOptimizer::MultistartOptimize() method in conjunction with an
  Optimizer object. This will probably remain the more common use case.

  But these notes discuss both to provide context for future extensions of both the local and global optimization techniques.

  In both cases, it is recommended to wrap Optimizer::Optimize() calls and/or MultistartOptimizer::MultistartOptimize()
  calls with code that sets up state, initial guesses, etc. for your specific problem.
  See usage in gpp_math and gpp_model_selection.

  At present, the an Optimizer class promises::

    template <typename ObjectiveFunctionEvaluator, typename DomainType>
    class Optimizer final {
      using ParameterStruct = OptimizerParameters;  // e.g., NewtonParameters
      Optimizer() = default;  // no state so default ctor ok

      int Optimize(const ObjectiveFunctionEvaluator& objective_evaluator, const ParameterStruct& parameters, const DomainType& domain, typename ObjectiveFunctionEvaluator::StateType * objective_state) const noexcept OL_NONNULL_POINTERS;
    }

  The Optimize() member function performs the desired optimization of the objective function specified through
  the (Evaluator, State) pair.  Most importantly, the initial guess will be read through
  objective_state::GetCurrentPoint().  Upon return, objective_state::GetCurrentPoint() will provide the final
  result, the  ``x`` for ``\argmax_x f(x)``, as determined by the optimization process.
  Note that since this is generally constrained optimization, in general ``f'(x) != 0`` (if the optima lies on a boundary).
  See specific Optimizer class docs (and ::Optimize() function docs) for details.

  Now, we outline specific optimization classes provided in thils file and the multistarting function that uses them:

  **3b, ii. OPTIMIZER CLASSES IN THIS FILE**

  class NullOptimizer<ObjectiveFunctionEvaluator, Domain>:
  NullOptimizer<...>::Optimize(...) (do nothing)

    * This optimizer does nothing; it provides an "identity" optimizer where Output := Input.
    * Its purpose is to allow MultistartOptimizer<...> to be used for 'dumb' searches.

  class GradientDescentOptimizer<ObjectiveFunctionEvaluator, Domain>:
  GradientDescentOptimizer<...>::Optimize(...) (restarted part of gradient descent)

    * Iteratively restarts GD from its previous ending point, unless convergence conditions are met
    * This calls:
      GradientDescentOptimization<ObjectiveFunctionEvaluator, Domain>()  (gradient descent)

      * Performs gradient descent to optimize specified objective function
      * Ensures (heuristically by modifying steps) that solutions remain in the specified domain
      * Calls out to ObjectiveFunctionEvaluator::ComputeObjectiveFunction() and ComputeGradObjectiveFunction()

  class NewtonOptimizer<ObjectiveFunctionEvaluator, Domain>:
  NewtonOptimizer<...>::Optimize() (Newton's method with refinement step)

    * First calls NewtonOptimization() to optimize.  Robustness heuristics are active to help make convergence easier.
    * Then calls NewtonOptimization() again with robustness heuristics off (heuristics should have converged fully
      or gotten us close) to ensure that convergence occurred.  (Sometimes the heuristics converge to nonsense;
      this second step will catch that.)
    * This function calls the following twice:
      NewtonOptimization<ObjectiveFunctionEvaluator, Domain>() (Newton's method for optimization)

      * Performs Newton iteration to optimize the templated objective function
      * Ensures (heuristically by modifying steps) that solutions remain in the specified domain
      * Calls out to ObjectiveFunctionEvaluator::ComputeObjectiveFunction(), ComputeGradObjectiveFunction(),
        and ComputeHessianObjectiveFunction()
      * Inner loop also calls ComputePLUFactorization() and PLUMatrixVectorSolve() from gpp_linear_algebra

   **3b, iii. MULTISTART OPTIMIZATION**
   class MultistartOptimizer<Optimizer<ObjectiveFunctionEvaluator, Domain> >:
   MultistartOptimizer<...>::MultistartOptimize() (multistarts any Optimizer from section 3b, ii.)

     * Calls Optimizer::Optimize() once for each point in the provided list of initial guesses
     * Multithreaded using OpenMP for performance
     * Reports the best result overall (and optionally each individual result)
     * Proxy for finding the global maximum since it is difficult/impossible to guarantee an optimum is global
       in general. See function comments (below) and header comments (above, 2c) for details.

     .. NOTE:: uses OptimizationIOContainer class (see declaration below for details) for inputting/outputting
        information about currently best-known objective values/points and the optimization result.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_HPP_

// #define OL_VERBOSE_PRINT

#if defined(OL_VERBOSE_PRINT) || 0  // always enable this if OL_VERBOSE_PRINT is defined
#define OL_OPTIMIZATION_VERBOSE_PRINT 1
#endif

#include <cmath>

#include <algorithm>
#include <exception>
#include <mutex>
#include <vector>

#include <omp.h>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

/*!\rst
  **Overview**

  When we ask openmp to parallelize a for loop, we can give it additional information on how to
  distribute the work. In particular, the overall work (N iterations) needs to be divided up amongst
  the threads. We have two major ways to affect how openmp structures the loop:
  ``schedule`` and ``chunk_size``.

  ``chunk_size`` changes meaning depending on ``schedule``. Here we list out the options for
  ``schedule`` as ``name (ENV_NAME, enum_name)`` where ``ENV_NAME`` is the corresponding value
  of ``OMP_SCHEDULE`` (not used by ``optimal_learning``) and ``enum_name`` is the corresponding
  type from ``opm_sched_t`` in ``omp.h``. Below, "work" refers to loop iterations (``N`` total).

  **Schedule Types**

  a. static ("static", omp_sched_static):

     Work is divided into ``N/chunk_size`` *contiguous* chunks (of ``chunk_size`` iterations) and
     distributed amongst the threads statically in a round-robin fashion. Use when you are
     confident all chunks will take the same amount of time.

     Low control overhead but high waste if one iteration is very slow (since the other
     threads will sit idle).

     Default ``chunk_size``: ``N / number_of_threads``.

     This schedule type is *repeatable*: repeated runs/calls (with the same work) will produce the
     same mapping of loop iterations to threads every time.

  b. dynamic ("dynamic", omp_sched_dynamic):

     Work is divided into ``N/chunk_size`` *contiguous* chunks (of ``chunk_size`` iterations) and
     distributed to threads as they complete their work, first-come first-serve. If there is
     a chunk that is very slow, the other threads can finish all remaining work instead of
     sitting idle.

     High control overhead, use when you have no idea how long each chunk will take.

     Default ``chunk_size``: 1.

     This schedule type does not produce repeatable mappings of iterations to threads.

  c. guided ("guided", omp_sched_guided):

     Work is divided into progressively smaller chunks; ``chunk_size`` sets the minimum value.
     As with dynamic, chunks are assigned on a first-come, first-serve basis.  Less overhead than
     dynamic (b/c ``chunk_size`` scale down).

     Useful when iteration times are similar but not identical.  Less overhead than dynamic while
     guaranteeing the waste case of static doesn't arise.

     Default ``chunk_size``: approximately ``N / number_of_threads``.

     This schedule type does not produce repeatable mappings of iterations to threads.

  d. auto ("auto", omp_sched_auto):

     The compiler decides how to map iterations to threads; this mapping is not required
     to be one of the previous choices.

     chunk_size has *no meaning* when the schedule is auto.
     See: https://gcc.gnu.org/onlinedocs/libgomp/omp_005fset_005fschedule.html

     This schedule type is not guaranteed to be repeatable.

  Further documentation:
  http://openmp.org/mp-documents/OpenMP3.1-CCard.pdf
  https://software.intel.com/en-us/articles/openmp-loop-scheduling
  http://publib.boulder.ibm.com/infocenter/comphelp/v8v101/index.jsp?topic=%2Fcom.ibm.xlcpp8a.doc%2Fcompiler%2Fref%2Fruompfor.htm
\endrst*/
struct ThreadSchedule {
  /*!\rst
    Construct a ThreadSchedule using the specified number of threads, schedule type, and chunk_size.

    \param
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      :schedule: static, dynamic, guided, or auto. See class comments for more details.
      :chunk_size: how to distribute work to threads; the precise meaning depends on schedule.
        Zero or negative chunk_size ask OpenMP to use its default behavior. See class comments for details.
  \endrst*/
  ThreadSchedule(int max_num_threads_in, omp_sched_t schedule_in, int chunk_size_in)
      : max_num_threads(max_num_threads_in), schedule(schedule_in), chunk_size(chunk_size_in) {
  }

  /*!\rst
    Construct a ThreadSchedule using the specified number of threads and schedule type with default chunk_size.

    \param
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
      :schedule: static, dynamic, guided, or auto. See class comments for more details.
  \endrst*/
  ThreadSchedule(int max_num_threads_in, omp_sched_t schedule_in) : ThreadSchedule(max_num_threads_in, schedule_in, 0) {
  }

  /*!\rst
    Construct a ThreadSchedule using the specified number of threads with default schedule type and chunk_size.

    \param
      :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
  \endrst*/
  explicit ThreadSchedule(int max_num_threads_in) : ThreadSchedule(max_num_threads_in, omp_sched_auto) {
  }

  /*!\rst
    Construct a ThreadSchedule using the default number of threads, schedule type, and chunk_size.
  \endrst*/
  ThreadSchedule() : ThreadSchedule(0) {
  }

  //! The maximum number of threads for use by OpenMP (generally should be <= # cores).
  //! The (default) value of 0 results in omp_get_num_procs() threads; note that this
  //! is limited by omp_get_thread_limit() (set in OMP_THREAD_LIMIT).
  int max_num_threads;

  //! The thread schedule to use: static, dynamic, guided, or auto. See class comments for more details.
  omp_sched_t schedule;

  //! Chunk size to use when distributing work to threads; the precise meaning depends on schedule.
  //! Zero or negative chunk_size ask OpenMP to use its default behavior. See class comments for details.
  int chunk_size;
};

/*!\rst
  This object holds the input/output fields for optimizers (maximization).  On input, this can be used to specify the current
  best known point (i.e., the optimizer will indicate no new optima found if it cannot beat this value).
  Upon completion, this struct should be read to determine the result of optimization.

  Since this object is used to communicate some inputs/outputs to the various optimization functions, any function
  using this object MUST obey its contract.

  The contract:
  On input, the optimizer will read best_objective_value_so_far.
  IF optimization results in a LARGER objective value, then:

    1. best_objective_value_so_far will be set to that new larger value
    2. best_point will be set to the point producing this new larger objective value
    3. found_flag will be SET to true

  ELSE:

    1. best_objective_value_so_far will be unmodified
    2. best_point will be unmodified
    3. found_flag will be SET to false

  The idea is for the user to be able to indicate what an improvement is.  For example, to optimize log likelihood as a
  function of hyperparameters, we could do:
  best_point = ``argmax_{x \in initial_guesses} f(x)``
  best_objective_value = ``f(best_point)``
  And then call multistart gradient descent (MGD).  Now, MGD will only change the best point/value if it converges to a better
  solution.  If convergence fails or MGD settles on a *worse* local maxima, found_flag will be SET to false, and the other
  fields of IOContainer will be *unmodified*.  If it finds a better solution, then found_flag will be SET to true and
  the other fields will report the new solution.
\endrst*/
struct OptimizationIOContainer final {
  /*!\rst
    Build an empty OptimizationIOContainer.  best_objective_value and best_point are initialized to zero; THIS MAY
    BE AN INVALID STATE.  See class docs for details.

    \param
      :problem_size: number of dimensions in the optimization problem (e.g., size of best_point)
  \endrst*/
  explicit OptimizationIOContainer(int problem_size_in)
      : problem_size(problem_size_in), best_objective_value_so_far(0.0), best_point(problem_size), found_flag(false) {
  }

  /*!\rst
    Build and fully initialize a OptimizationIOContainer.  See class docs for details.

    \param
      :problem_size: number of dimensions in the optimization problem (e.g., size of best_point)
      :best_objective_value: the best objective function value seen so far
      :best_point: the point to associate with best_objective_value
  \endrst*/
  OptimizationIOContainer(int problem_size_in, double best_objective_value, double const * restrict best_point_in)
      : problem_size(problem_size_in),
        best_objective_value_so_far(best_objective_value),
        best_point(best_point_in, best_point_in + problem_size),
        found_flag(false) {
  }

  OptimizationIOContainer(OptimizationIOContainer&& OL_UNUSED(other)) = default;

  //! spatial dimension (e.g., dimensions of a point in points_sampled, num_hyperparameters)
  const int problem_size;
  //! the best objective function value seen
  double best_objective_value_so_far;
  //! the point producing ``best_objective_value_so_far`` after successful optimizzation (``found_flag = true``);
  //! otherwise it contains the original, unmodified values from when the function was called
  std::vector<double> best_point;
  //! true if the optimizer found improvement
  bool found_flag;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(OptimizationIOContainer);
};

/*!\rst
  TODO(GH-390): Implement Polyak-Ruppert Averaging for Gradient Descent

  Implements gradient-descrent to to find a locally optimal (maximal here) value of the specified objective function.
  Additional high-level discussion is provided in section 2a) in the header docs of this file.

  Basic gradient descent (GD) to optimize objective function ``f(x)``::

    input: initial_guess

    next_point = initial_guess
    i = 0;
    while (not converged) {
      direction = derivative of f(x) at next_point
      step_scale = compute step_size scaling: pre_mult * (i+1)^(-gamma)

      next_point += step_scale * direction
      ++i
    }

  So it marches along the direction of largest gradient (so the steepest descent) for some distance.  The distance
  is a combination of the size of the gradient and the step_scale factor.  Here, we use an exponentially decreasing
  scale to request progressively smaller step sizes: ``(i+1)^(-gamma)``, where ``i`` is the iteration number

  We do not allow the step to take next_point out of the domain; if this happens, the update is limited.
  Thus the solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  We may also limit very large updates (controlled via max_relative_change).  Decreasing this value
  makes gradient descent (GD) more stable but also slower.  For very sensitive problems like hyperparameter
  optimization, max_relative_change = 0.02 is suggested; for less sensitive problems
  (e.g., EI, especially analytic), you can use 1.0 (or near).

  The constraint implementation (no stepping outside the domain) and the large update limiting are not "pure" gradient
  descent approaches.  They are all heuristics meant to improve Newton's robustness.  The constraint implementation
  in particular may lead to non-convergence and it also may not find constrained optima that lie exactly on a boundary.  This would
  require a more general handling where we search in an ``d-1`` dimensional subspace (i.e., only on the boundary).

  Note that we are using an absolute tolerance here, based on the size of the most recent step.
  The suggested value is 1.0e-7, although this may need to be loosened for problems with 'difficult' optima (e.g., the shape
  is not locally very peaked).  Setting too high of a tolerance can cause wrong answers--e.g., we stop at a point
  that is not an optima but simply an region with small gradient.  Setting the tolerance too low may make convergence impossible;
  GD could get stuck (bouncing between the same few points) or numerical effects could make it impossible to satisfy tolerance.

  Finally, GD terminates if updates are very small.

  .. Note:: in general, you should not call/instantiate this function directly.  Instead, create a GradientDescentOptimizer object
         and call its ::Optimize() function.

  problem_size refers to objective_state->GetProblemSize(), the number of dimensions in a "point" aka the number of
  variables being optimized.  (This might be the spatial dimension for EI or the number of hyperparameters for log likelihood.)

  .. Note:: these comments are are copied to GradientDescentOptimizer.optimize() in python/python_version/optimization.py.

  \param
    :objective_evaluator: reference to object that can compute the objective function and its gradient
    :gd_parameters: GradientDescentParameters object that describes the parameters controlling gradient descent optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :objective_state[1]: a properly configured state object for the ObjectiveFunctionEvaluator template parameter
                         objective_state.GetCurrentPoint() will be used to obtain the initial guess
  \output
    :objective_state[1]: a state object whose temporary data members may have been modified
                         objective_state.GetCurrentPoint() will return the point yielding the best objective function value
                         according to gradient descent
\endrst*/
template <typename ObjectiveFunctionEvaluator, typename DomainType>
OL_NONNULL_POINTERS void GradientDescentOptimization(
    const ObjectiveFunctionEvaluator& objective_evaluator,
    const GradientDescentParameters& gd_parameters,
    const DomainType& domain,
    typename ObjectiveFunctionEvaluator::StateType * objective_state) {
  const int problem_size = objective_state->GetProblemSize();
  std::vector<double> grad_objective(problem_size);
  std::vector<double> step(problem_size);
  std::vector<double> next_point(problem_size);

  // read out starting point coordinates
  objective_state->GetCurrentPoint(next_point.data());

  // save off some data for reporting if needed
#ifdef OL_VERBOSE_PRINT
  std::vector<double> initial_point(problem_size);
  // initial value of the objective function
  double obj_func_initial = objective_evaluator.ComputeObjectiveFunction(objective_state);
  std::copy(next_point.begin(), next_point.end(), initial_point.begin());
#endif

  const double step_tolerance = gd_parameters.tolerance / static_cast<double>(gd_parameters.max_num_steps);
  for (int i = 0; i < gd_parameters.max_num_steps; ++i) {
    double alpha_n = gd_parameters.pre_mult*std::pow(static_cast<double>(i+1), -gd_parameters.gamma);
    objective_evaluator.ComputeGradObjectiveFunction(objective_state, grad_objective.data());
#ifdef OL_VERBOSE_PRINT
    if (i == 0) {
      OL_VERBOSE_PRINTF("objective fcn gradients, pre: ");
      PrintMatrix(grad_objective.data(), 1, problem_size);
    }
#endif

    // set up desired step size
    for (int j = 0; j < problem_size; ++j) {
      step[j] = alpha_n*grad_objective[j];
    }
    // limit step size to ensure we stay inside the domain
    domain.LimitUpdate(gd_parameters.max_relative_change, next_point.data(), step.data());
    // take the step
    for (int j = 0; j < problem_size; ++j) {
      next_point[j] += step[j];
    }

    // update state
    objective_state->SetCurrentPoint(objective_evaluator, next_point.data());

    double norm_step = VectorNorm(step.data(), problem_size);
    if (norm_step < step_tolerance) {
      ++i;
      break;
    }
  }  // end loop over i (gradient descent)

  OL_VERBOSE_PRINTF("Coord index: Initial    :        Final values      :        Difference        :\n");
  for (int j = 0; j < problem_size; ++j) {
    OL_VERBOSE_PRINTF("%d: %.18E : %.18E : %.18E\n", j, initial_point[j], next_point[j], next_point[j] - initial_point[j]);
  }

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
  // final value of the objective function
  double obj_func_final = objective_evaluator.ComputeObjectiveFunction(objective_state);

  OL_VERBOSE_PRINTF("Initial objective fcn value: %.18E, final objective fcn value: %.18E\n", obj_func_initial, obj_func_final);

  // relative change from the initial to final values of the objective function
  double obj_func_relative_change = std::fabs((obj_func_final - obj_func_initial)/obj_func_initial);
  double obj_func_absolute_change = std::fabs(obj_func_final - obj_func_initial);
  if (obj_func_final < obj_func_initial && obj_func_relative_change > gd_parameters.tolerance && obj_func_absolute_change > 1.0e-20) {
    OL_VERBOSE_PRINTF("ERROR: objective fcn got worse!!! |Difference|: %.18E, |Relative|: %.18E\n", obj_func_absolute_change, obj_func_relative_change);
  }
#endif

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
  for (int j = 0; j < problem_size; ++j) {
    if (std::fabs(next_point[j] - domain[2*j + 0]) < 1.0e-8 || std::fabs(next_point[j] - domain[2*j + 1]) < 1.0e-8) {
      OL_VERBOSE_PRINTF("WARNING: coord %d is very close to boundaries! coord = %.18E, lower boundary = %.18E, upper boundary = %.18E\n", j, next_point[j], domain[2*j+0], domain[2*j+1]);
    }
  }
#endif

#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("objective fcn gradients, post: ");
  PrintMatrix(grad_objective.data(), 1, problem_size);
#endif
}

/*!\rst
  Uses Newton's Method to optimize the value of an objective function, f (e.g., log marginal likelihood).  Newton's method is
  a root-finding technique, so for optimization, we are searching for points where gradient = 0.
  http://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
  has some basic details.  Additional high-level discussion is provided in section 2b) in the header docs of this file.

  Each newton step is given by:
  ``\theta_{n+1} = \theta_n - \bar{H}_f^-1(\theta_n) * \nabla f(\theta_n)``,
  where ``\bar{H} = H_f(\theta_n) - 1/time_factor * I``, and
  ``\nabla f`` and ``H`` are the gradient and Hessian of the objective function, respectively,
  ``time_factor = time_factor_0 * gamma^n``, and ``I`` is the identity matrix.

  This method terminates early if a possible solution is found--``||\nabla f||`` is sufficiently small.

  Choosing a small ``gamma`` (e.g., ``1.0 < gamma <= 1.01``) and ``time_factor`` (e.g., ``0 < time_factor <= 1.0e-3``)
  leads to more consistent/stable convergence at the cost of slower performance (and in fact
  for gamma or time_factor too small, gradient descent is preferred).  Conversely, choosing more
  aggressive values may lead to very fast convergence at the cost of more cases failing to
  converge.

  ``gamma = 1.01, time_factor = 1.0e-3`` should lead to good robustness at reasonable speed.  This should be a fairly safe default.
  ``gamma = 1.05, time_factor = 1.0e-1`` will be several times faster but not as robust.

  Notice that we modify the Hessian in an attempt to improve the region of "attraction"
  for Newton.  To do this, we add diagonal dominance to the Hessian: ``\bar{H} = H - 1/time_factor * I``.
  In the classic/standard version of Newton's Method, ``time_factor = \infty`` so ``\bar{H}`` is just the Hessian.
  Note: we subtract because we are maximizing the objective and ``H`` is strictly negative definite at a maxima.

  When ``\theta_i`` is far away from ``\theta_{opt}``, having a very small time_factor causes the Newton
  update to behave like the Gradient Descent update.  This is because ``H - LARGE_VALUE * I`` makes
  ``H`` "look like" a scaled identity matrix (with a relatively tiny amount of noise added).  Thus
  using very large time_factor for many iterations is inefficient: the udpates are gradient-descent-like,
  but the cost is 5-7x more (in this case, not in general).  But hoping that
  gradient descent-like steps guide us toward into a convergence region for Newton, time_factor
  is increased (by a constant factor) each iteration.  As ``time_Factor -> \infinity``, ``\bar{H} -> H``;
  hence we recover Newton's fast convergence properties.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This means a grid will probably be necessary
  to give a good enough initial guess.

  Guaranteed execute AT MOST max_num_restarts newton steps.

  We also allow an under/over-relaxation factor, allowing the update to be scaled up/down.

  Thus the basic structure for optimizing f(\theta) is::

    \theta_i = initial guess
    for i = 0:max_iterations {
      compute gradient of f: \nabla f(\theta_i)
      if (||gradient f|| < tolerance) exit
      compute hessian of f: H(\theta_i)

      time_factor = initial * growth_factor^i
      modify hessian: \bar{H} = H - 1/time_factor * I (I is identity matrix)

      compute update: update = \bar{H}^-1 * \nabla f  (performed without forming inverse)

      relax update: update *= relaxation_factor

      apply update: \theta_i = \theta_i - update
    }

  We do not allow the step to take next_point out of the domain; if this happens, the update is limited.  For now this limiting
  is done in a naive way; for example we may restrict the step size to 50% of the distance to the nearest boundary.

  We may also limit very large updates (controlled via max_relative_change).  For very sensitive problems like hyperparameter
  optimization, max_relative_change = 0.02 is suggested; for less sensitive problems (e.g., EI, especially analytic),
  you can use 1.0.

  The constraint implementation (no stepping outside the domain), the large update limiting, and the hessian modification
  are not "pure" Newton approaches.  They are all heuristics meant to improve Newton's robustness.  The constraint implementation
  in particular may lead to non-convergence and it also may not find constrained optima that lie exactly on a boundary.  This would
  require a more general handling where we search in an d-1 dimensional subspace (i.e., only on the boundary).

  Finally, note that we are using an absolute tolerance here, based on magnitude of the gradient, not step distance!  (Contrast
  this with tolerance in gradient descent.)
  The suggested value is 1.0e-13, although this may need to be
  loosened for ill-conditioned problems.  Setting too high of a tolerance can cause wrong answers--e.g., we stop at a point
  that is not an optima but simply an region with small gradient.  Setting the tolerance too low will make convergence impossible
  due to loss of accuracy through numerical effects.

  TODO(GH-161): Investigate/add stagnation detection to newton, so it stops when going too many steps without improving the result.

  TODO(GH-133): (GH-134) Improve Newton's performance/robustness.

  .. WARNING:: this method does not check the eigenvalues of H at this solution to verify that it is an optima and not a saddle
      or an indeterminate result.
      TODO(GH-121): Add optima classification to Newton.

  Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. Note:: in general, you should not call/instantiate this function directly.  Instead, create a NewtonOptimizer object
         and call its ::Optimize() function.

  problem_size refers to objective_state->GetProblemSize(), the number of dimensions in a "point" aka the number of
  variables being optimized.  (This might be the spatial dimension for EI or the number of hyperparameters for log likelihood.)

  \param
    :objective_evaluator: reference to object that can compute the objective function and its gradient
    :newton_parameters: NewtonParameters object that describes the parameters newton optimization
      (e.g., number of iterations, tolerances, additional diagonal dominance)
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :objective_state[1]: a properly configured state object for the ObjectiveFunctionEvaluator template parameter
                         objective_state.GetCurrentPoint() will be used to obtain the initial guess
  \output
    :objective_state[1]: a state object whose temporary data members may have been modified
                         objective_state.GetCurrentPoint() will return the point yielding the best objective function value
                         according to newton
  \return
    number of errors
\endrst*/
template <typename ObjectiveFunctionEvaluator, typename DomainType>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int NewtonOptimization(
    const ObjectiveFunctionEvaluator& objective_evaluator,
    const NewtonParameters& newton_parameters,
    const DomainType& domain,
    typename ObjectiveFunctionEvaluator::StateType * objective_state) {
  if (unlikely(newton_parameters.max_num_restarts <= 0)) {
    return 0;
  }
  const int problem_size = objective_state->GetProblemSize();
  std::vector<double> next_point(problem_size);

  // read out starting point coordinates from state
  objective_state->GetCurrentPoint(next_point.data());

  std::vector<int> pivot(problem_size);
  std::vector<double> step(problem_size);
  std::vector<double> gradient_objective(problem_size);
  std::vector<double> hessian_objective(Square(problem_size));

  // const double relaxation_factor = 1.0;  // under/over-relxation of newton updates
  const double step_tolerance = newton_parameters.tolerance / static_cast<double>(newton_parameters.max_num_steps*10);
  // computing iterative updates \theta_{n+1} = \theta_n - H_f^-1(\theta_n) * \nabla f(\theta_n)
  // where f is the objective function, and \nabla f and H are its gradient and Hessian, respectively
  double time_factor = newton_parameters.time_factor;
  int error = 0;
  int newton_iter;  // track the number of newton iterations
  for (newton_iter = 0; newton_iter < newton_parameters.max_num_steps; ++newton_iter) {
    objective_evaluator.ComputeGradObjectiveFunction(objective_state, gradient_objective.data());

    double norm_gradient_objective = VectorNorm(gradient_objective.data(), problem_size);
#ifdef OL_VERBOSE_PRINT
    OL_VERBOSE_PRINTF("iter %d: CFL: %.18E, objective fcn: %.18E, norm gradient: %.18E\n", newton_iter, time_factor, objective_evaluator.ComputeObjectiveFunction(objective_state), norm_gradient_objective);
    PrintMatrix(gradient_objective.data(), 1, problem_size);
    OL_VERBOSE_PRINTF("iter %d: norm gradient: %.18E\n", newton_iter, norm_gradient_objective);
#endif
    // if (unlikely(norm_gradient_objective <= tolerance && newton_iter > 0)) {
    //   break;  // coordinates are no longer changing notably, so stop
    // }
    if (unlikely(norm_gradient_objective <= newton_parameters.tolerance)) {
      break;  // coordinates are no longer changing notably, so stop
    }

    objective_evaluator.ComputeHessianObjectiveFunction(objective_state, hessian_objective.data());

    // add diagonal dominance to the Hessian
    for (int j = 0; j < problem_size; ++j) {
      hessian_objective[j*problem_size + j] -= 1.0/time_factor;
    }
    // reduce amount of diagonal dominance to be added next iteration
    time_factor *= newton_parameters.gamma;
    // TODO(GH-134): update limiting.  If the update change is too large (factor of 2?), then consider:
    // 1) limiting the update
    // 2) and/or DECREASING time_factor to take more conservative steps
    // 3) line search to find a better update (instead of just using max step size limiting)
    // this might allow for more aggressive gamma values.  we'd automatically hold time_factor small until things are good
    // and then launch up to large time_factor values quickly

    // The issue is that currently, time_factor finishes around 10 or 100, which is too small to be sure of convergence; this
    // requires a secondary Newton run starting at the "converged" location with time_factor = 1e30 (huge) to double check it

#ifdef OL_VERBOSE_PRINT
    PrintMatrix(hessian_objective.data(), problem_size, problem_size);
    OL_VERBOSE_PRINTF("\n");
#endif
    // PLU-factor the Hessian and compute the Newton update vector by solving the system of eqns:
    // H_f(\theta_n) * update_n = \nabla f(\theta_n)
    error = ComputePLUFactorization(problem_size, pivot.data(), hessian_objective.data());
#ifdef OL_VERBOSE_PRINT
    PrintMatrix(hessian_objective.data(), problem_size, problem_size);
#endif
    if (unlikely(error != 0)) {
      break;  // system is singular, stop
    }
    PLUMatrixVectorSolve(problem_size, hessian_objective.data(), pivot.data(), gradient_objective.data());

    // set up desired step size
    for (int j = 0; j < problem_size; ++j) {
      step[j] = -gradient_objective[j];
    }
    // limit step size to ensure we stay inside the domain
    domain.LimitUpdate(newton_parameters.max_relative_change, next_point.data(), step.data());
    // take the step
    for (int j = 0; j < problem_size; ++j) {
      next_point[j] += step[j];
    }

    // set new point for next run
    objective_state->SetCurrentPoint(objective_evaluator, next_point.data());
#ifdef OL_VERBOSE_PRINT
    norm_gradient_objective = VectorNorm(gradient_objective.data(), problem_size);
    OL_VERBOSE_PRINTF("iter %d: norm update: %.18E, coord:\n", newton_iter, norm_gradient_objective);
    PrintMatrix(next_point.data(), 1, problem_size);
#endif

    double norm_step = VectorNorm(step.data(), problem_size);
    if (norm_step < step_tolerance) {
      ++newton_iter;
      break;
    }
  }  // end loop over newton_iter

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
  double norm_gradient_objective = VectorNorm(gradient_objective.data(), problem_size);
  OL_VERBOSE_PRINTF("iter %d: norm gradient: %.18E\n", newton_iter, norm_gradient_objective);
  PrintMatrix(next_point.data(), 1, problem_size);
#endif

  return error;
}

/*!\rst
  The "null" or identity optimizer: it does nothing, giving the same output its inputs
  This is useful to allow the multistart optimizer template to be reused for 'dumb' searches and
  nontrivial optimization.  In the former, we just need to evaluate the objective at each initial guess,
  so there is no optimization to be done at each point (hence null optimizer).  In the latter,
  we kick off an optimization run (e.g., gradient descent, newton) at each initial guess.
\endrst*/
template <typename ObjectiveFunctionEvaluator_, typename DomainType_>
class NullOptimizer final {
 public:
  using ObjectiveFunctionEvaluator = ObjectiveFunctionEvaluator_;
  using DomainType = DomainType_;
  using ParameterStruct = NullParameters;

  NullOptimizer() = default;

  /*!\rst
    Perform a null optimization: this does nothing.

    \param
      :objective_state[1]: a properly configured state object for the ObjectiveFunctionEvaluator template parameter
                           objective_state.GetCurrentPoint() will be used to obtain the initial guess
    \output
      :objective_state[1]: a state object whose temporary data members may have been modified
                           objective_state.GetCurrentPoint() will return the point as the intial guess
    \return
      number of errors, always 0
  \endrst*/
  int Optimize(const ObjectiveFunctionEvaluator& OL_UNUSED(objective_evaluator),
               const ParameterStruct& OL_UNUSED(parameters), const DomainType& OL_UNUSED(domain),
               typename ObjectiveFunctionEvaluator::StateType * OL_UNUSED(objective_state))
      const noexcept OL_NONNULL_POINTERS OL_PURE_FUNCTION {
    return 0;
  }

  OL_DISALLOW_COPY_AND_ASSIGN(NullOptimizer);
};

/*!\rst
  Gradient descent (GD) optimization.  This class optimizes using restarted GD (see comments on the Optimize()) function.
\endrst*/
template <typename ObjectiveFunctionEvaluator_, typename DomainType_>
class GradientDescentOptimizer final {
 public:
  using ObjectiveFunctionEvaluator = ObjectiveFunctionEvaluator_;
  using DomainType = DomainType_;
  using ParameterStruct = GradientDescentParameters;

  GradientDescentOptimizer() = default;

  /*!\rst
    Optimize a given objective function (represented by ObjectiveFunctionEvaluator; see file comments for what this must provide)
    using restarted gradient descent (GD).

    See section 2a) and 3b, i) in the header docs and the docs for GradientDescentOptimization() for more details.

    Guaranteed to call GradientDescentOptimization() AT MOST max_num_restarts times.
    GradientDescentOptimization() implements gradient descent; see function comments above for details.
    This method calls gradient descent, then restarts (by calling GD again) from the GD's result point.  This is done until
    max_num_restarts is reached or the result point stops changing (compared to tolerance).

    Note that we are using an absolute tolerance, based on the size of the most recent step\*.  Here, 'step' is the
    distance covered by the last restart, not the last GD iteration (as in GradientDescentOptimization()).
    The suggested value is 1.0e-7, although this may need to be loosened for problems with 'difficult' optima (e.g., the shape
    is not locally very peaked).  Setting too high of a tolerance can cause wrong answers--e.g., we stop at a point
    that is not an optima but simply an region with small gradient.  Setting the tolerance too low may make convergence impossible;
    GD could get stuck (bouncing between the same few points) or numerical effects could make it impossible to satisfy tolerance.

    \* As opposed to say based on changes in the objective function.

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    true optima (i.e., the gradient may be substantially nonzero).

    problem_size refers to objective_state->GetProblemSize(), the number of dimensions in a "point" aka the number of
    variables being optimized.  (This might be the spatial dimension for EI or the number of hyperparameters for log likelihood.)

    \param
      :objective_evaluator: reference to object that can compute the objective function and its gradient
      :gd_parameters: GradientDescentParameters object that describes the parameters controlling gradient descent optimization
        (e.g., number of iterations, tolerances, learning rate)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :objective_state[1]: a properly configured state object for the ObjectiveFunctionEvaluator template parameter
                           objective_state.GetCurrentPoint() will be used to obtain the initial guess
    \output
      :objective_state[1]: a state object whose temporary data members may have been modified
                           objective_state.GetCurrentPoint() will return the point yielding the best objective function value
                           according to gradient descent
    \return
      number of errors, always 0
  \endrst*/
  int Optimize(const ObjectiveFunctionEvaluator& objective_evaluator, const ParameterStruct& gd_parameters,
               const DomainType& domain, typename ObjectiveFunctionEvaluator::StateType * objective_state)
      const OL_NONNULL_POINTERS {
    if (unlikely(gd_parameters.max_num_restarts <= 0)) {
      return 0;
    }
    const int problem_size = objective_state->GetProblemSize();
    std::vector<double> current_point(problem_size);
    std::vector<double> next_point(problem_size);

    // loop structure expects that "next_point" contains the new current location at the start of each iteration
    objective_state->GetCurrentPoint(next_point.data());

    for (int i = 0; i < gd_parameters.max_num_restarts; ++i) {
      // save off current location so we can compute the update norm
      std::copy(next_point.begin(), next_point.end(), current_point.begin());
      // get next gradient descent update
      GradientDescentOptimization(objective_evaluator, gd_parameters, domain, objective_state);
      objective_state->GetCurrentPoint(next_point.data());

      // compute norm of the update
      for (int j = 0; j < problem_size; ++j) {
        current_point[j] -= next_point[j];
      }
      double norm_delta_coord = VectorNorm(current_point.data(), problem_size);
      OL_VERBOSE_PRINTF("norm of coord change: %.18E\n", norm_delta_coord);
      OL_VERBOSE_PRINTF("^Step %d^\n", i+1);

      if (norm_delta_coord <= gd_parameters.tolerance) {
        break;  // point are no longer changing notably, so stop
      }
    }

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
    if (norm_delta_coord > gd_parameters.tolerance) {
      // we didn't converge to a sufficient degree
      OL_VERBOSE_PRINTF("WARNING: gradient descent may not be fully converged yet!  In the call, the point changed by (RMS): %.18E\n", norm_delta_coord);
    }
#endif

    return 0;
  }

  OL_DISALLOW_COPY_AND_ASSIGN(GradientDescentOptimizer);
};

/*!\rst
  Newton optimization.  This class optimizes using Newton's method with a refinement step (see comments on the Optimize()) function.
\endrst*/
template <typename ObjectiveFunctionEvaluator_, typename DomainType_>
class NewtonOptimizer final {
 public:
  using ObjectiveFunctionEvaluator = ObjectiveFunctionEvaluator_;
  using DomainType = DomainType_;
  using ParameterStruct = NewtonParameters;

  NewtonOptimizer() = default;

  /*!\rst
    Uses Newton's Method to optimize the value of an objective function, f (e.g., log marginal likelihood).

    .. NOTE:: this function wraps NewtonOptimization(), see above.  It first calls that function directly, then calls it again
        with a modified newton_parameters struct: the param struct is modified to run newton with a small number
        of iterations at a huge time_factor (to remove the diagonal dominance adjustment entirely).  We do this
        to ensure that Newton has converged.

    See section 2b) and 3b, ii) in the header docs and the docs for NewtonOptimization() for more details.

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    true optima (i.e., the gradient may be substantially nonzero).

    problem_size refers to objective_state->GetProblemSize(), the number of dimensions in a "point" aka the number of
    variables being optimized.  (This might be the spatial dimension for EI or the number of hyperparameters for log likelihood.)

    \param
      :objective_evaluator: reference to object that can compute the objective function and its gradient
      :newton_parameters: NewtonParameters object that describes the parameters newton optimization
        (e.g., number of iterations, tolerances, additional diagonal dominance)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :objective_state[1]: a properly configured state object for the ObjectiveFunctionEvaluator template parameter
                           objective_state.GetCurrentPoint() will be used to obtain the initial guess
    \output
      :objective_state[1]: a state object whose temporary data members may have been modified
                          objective_state.GetCurrentPoint() will return the point yielding the best objective function value
                          according to newton
    \return
      number of errors
  \endrst*/
  int Optimize(const ObjectiveFunctionEvaluator& objective_evaluator, const ParameterStruct& newton_parameters,
               const DomainType& domain, typename ObjectiveFunctionEvaluator::StateType * objective_state)
      const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    int total_errors = 0;

    total_errors += NewtonOptimization(objective_evaluator, newton_parameters, domain, objective_state);

    // TODO(GH-174): If newton_parameters becomes a class member, so should this refinement version.
    const int max_num_steps_refinement = 10;  // max number of newton steps; don't need many here b/c it should already be converged
    const double time_factor_refinement = 1.0e40;  // scaling factor high enough to remove diagonal dominance adjustment
    ParameterStruct newton_parameters_refinement(
        1,
        max_num_steps_refinement,
        newton_parameters.gamma,
        time_factor_refinement, newton_parameters.max_relative_change,
        newton_parameters.tolerance);

    total_errors += NewtonOptimization(objective_evaluator, newton_parameters_refinement, domain, objective_state);
    return total_errors;
  }

  OL_DISALLOW_COPY_AND_ASSIGN(NewtonOptimizer);
};

/*!\rst
  This is a general, template class for multistart optimization.  It is designed to be used with the various Optimizer
  classes in this file (e.g., NullOptimizer, GradientDescentOptimizer, NewtonOptimizer).  The multistart process is
  multithreaded using OpenMP so that we can start from multiple initial guesses across multiple threads simultaneously.
  See section 2c) and 3b, iii) in the header docs at the top of the file for more details.

  The use with GradientDescentOptimizer, NewtonOptimizer, etc. are standard practice in nonlinear optimization.  In particular,
  without special properties like convexity, single-start optimizers can converge to local optima.  In general, a nonlinear
  function can have many local optima, so the only way to improve\* your chances of finding the global optimum is to start
  from many different locations.  This will be the typical use case for MultistartOptimizer<...>::MultistartOptimize().

  \* Improve is intentional here.  In the general case, you are not *guaranteed* (in finite time) to find the global optimum.

  Use with NullOptimizer requires special mention here as it might seem silly. This case reduces to evaluating the
  objective function at every point of initial_guesses.  Through function_values, you can get the objective value at each
  of point of initial_guesses too (e.g., for plotting).  So use MultistartOptimize with NullOptimzer to perform a
  'dumb' search (e.g., initial_guesses can be obtained from a grid, random sampling, etc.).  NullOptimizer allows 'dumb' search
  to use the same code as multistart optimization.  'Dumb' search is inaccurate but it never fails, so we often use it as a
  fall-back when more advanced (e.g., gradient descent) techniques fail.

  This class provides just one method (for now), MultistartOptimize(); see below.

  .. Note:: comments copied to MultistartOptimizer in python_version/optimization.py.
\endrst*/
template <typename Optimizer_>
class MultistartOptimizer final {
 public:
  using Optimizer = Optimizer_;
  using ObjectiveFunctionEvaluator = typename Optimizer::ObjectiveFunctionEvaluator;
  using DomainType = typename Optimizer::DomainType;
  using ParameterStruct = typename Optimizer::ParameterStruct;

  MultistartOptimizer() = default;

  /*!\rst
    Performs multistart optimization with the specified Optimizer (class template parameter)
    to optimize the specified ObjectiveFunctionEvaluator over the specified DomainType.
    Optimizer behavior is controlled by the specified ParameterStruct. See class docs and header
    docs of this file, section 2c and 3b, iii), for more information.

    The method allows you to specify what the current best is, so that if optimization cannot
    beat it, no improvement will be reported.  It will otherwise report the overall best
    improvement (through io_container) as well as the result of every individual multistart run
    if desired (through function_values).

    .. Note:: comments copied to MultistartOptimizer.optimize() in python_version/optimization.py.

    Generally, you will not call this function directly.  Instead, it is intended to be used in
    wrappers that set up state, thread_schedule, etc. for the specific optimization problem at hand.
    For examples with Expected Improvement (EI), see gpp_math:

    * ``EvaluateEIAtPointList()``
    * ``ComputeOptimalPointsToSampleViaMultistartGradientDescent()``

    or gpp_model_selection:

    * ``EvaluateLogLikelihoodAtPointList()``
    * ``MultistartGradientDescentHyperparameterOptimization()``
    * ``MultistartNewtonHyperparameterOptimization()``

    problem_size refers to objective_state->GetProblemSize(), the number of dimensions in a "point"
    aka the number of variables being optimized.  (This might be the spatial dimension for EI or the
    number of hyperparameters for log likelihood.)

    \param
      :optimizer: object with the desired Optimize() functionality (e.g., do nothing for
        'dumb' search, gradient descent, etc.)
      :objective_evaluator: reference to object that can compute the objective function,
        its gradient, and/or its hessian, depending on the needs of optimizer
      :optimizer_parameters: Optimizer::ParameterStruct object that describes the parameters
        for optimization (e.g., number of iterations, tolerances, scale factors, etc.)
      :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
      :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e.,
        max_num_threads, schedule type, chunk_size
      :initial_guesses[problem_size][num_multistarts]: list of points at which to start
        optimization runs; all points must lie INSIDE the specified domain
      :num_multistarts: number of random points to use from initial guesses
      :objective_state_vector[thread_schedule.max_num_threads]:
        properly constructed/configured ObjectiveFunctionEvaluator::State objects,
        at least one per thread objective_state.GetCurrentPoint() will be used to obtain the initial guess
      :io_container[1]: object with best_objective_value_so_far and corresponding best_point properly initialized.
                        See struct docs in gpp_optimization.hpp for details.
    \output
      :objective_state_vector[thread_schedule.max_num_threads]: internal states of state objects may be modified
      :function_values[num_multistarts]: objective fcn value at the end of each
        optimization run, in the same order as initial_guesses. Can be used to check
        what each optimization run converged to.
        More commonly used only with NullOptimizer to get a list of objective values  at each point of initial_guesses.
        Never dereferenced if nullptr.
      :io_container[1]: object container new best_objective_value_so_far and corresponding
        best_point IF found_flag is true.
        Unchanged from input otherwise. See struct docs in gpp_optimization.hpp for details.
    \raise
      if any of objective_state_vector->SetCurrentPoint(), optimizer.Optimize(), or
      objective_evaluator.ComputeObjectiveFunction() throws, the exception (or one of the exceptions in the
      event of multiple throws due to threading, usually the first temporally) will be saved and rethrown by
      this function. ``io_container`` will be in a valid state; ``function_values`` may not.
  \endrst*/
  void MultistartOptimize(const Optimizer& optimizer, const ObjectiveFunctionEvaluator& objective_evaluator,
                          const ParameterStruct& optimizer_parameters, const DomainType& domain,
                          const ThreadSchedule& thread_schedule, double const * restrict initial_guesses,
                          int num_multistarts,
                          typename ObjectiveFunctionEvaluator::StateType * objective_state_vector,
                          double * restrict function_values, OptimizationIOContainer * restrict io_container) {
    const int problem_size = objective_state_vector[0].GetProblemSize();

    // exception_capture_flag "guards" captured_exception. std::called_once() guarantees that will only execute
    // any of its Callable(s) ONCE for each unique std::once_flag. See C++11 Standard Library documentation (``<mutex>``).
    // These tools together ensure that we can capture exceptions from OpenMP parallel regions in a thread-safe way.
    std::once_flag exception_capture_flag;
    // pointer-like object that manages an exception captured with std::capture_exception(). We use this to capture
    // exceptions thrown from the OpenMP parallel region.
    // See the try-catch block in the ``#pragma omp for`` region for more information.
    std::exception_ptr captured_exception;

    io_container->found_flag = false;
    const double best_objective_value_so_far_init = io_container->best_objective_value_so_far;
    int total_errors = 0;

    omp_set_schedule(thread_schedule.schedule, thread_schedule.chunk_size);
#pragma omp parallel num_threads(thread_schedule.max_num_threads)
    {
      double best_objective_value_so_far_local = best_objective_value_so_far_init;
      double objective_value;
      std::vector<double> next_point_local(problem_size);
      std::vector<double> best_next_point_local(problem_size);
      int thread_id = omp_get_thread_num();

#pragma omp for nowait schedule(runtime) reduction(+:total_errors)
      for (int i = 0; i < num_multistarts; ++i) {
        // It is illegal for exceptions to leave OpenMP blocks. Violating this condition leads to undefined behavior
        // (usually program termination). See:
        // http://www.thinkingparallel.com/2006/11/30/making-exceptions-work-with-openmp-some-tiny-workarounds/
        // As noted in the spec:
        //   "A 'structured block' is a single statement or a compound statement with a single entry at the top and a
        //   single exit at the bottom."
        //   http://www.openmp.org/mp-documents/OpenMP3.0-SummarySpec.pdf
        // Exceptions, break, and other control-flow modifying statements violate the single exit condition by allowing
        // execution to leave the block on a different path (e.g., catch statement).

        // Thus, we must catch and handle *all* exceptions within this ``omp for`` region. To propagate an
        // exception out of this structured block, we will capture an active exception into a std::exception_ptr.
        // Typically, the *first* exception thrown (temporally) will be captured.
        try {
          objective_state_vector[thread_id].SetCurrentPoint(objective_evaluator, initial_guesses + i*problem_size);

          if (unlikely(optimizer.Optimize(objective_evaluator, optimizer_parameters, domain, objective_state_vector + thread_id) != 0)) {
            ++total_errors;
          }

          // compute objective at the new potential optimum; note Optimize() guarantees optimum point is already in state
          objective_value = objective_evaluator.ComputeObjectiveFunction(objective_state_vector + thread_id);

          if (unlikely(function_values != nullptr)) {
            function_values[i] = objective_value;
          }

          // update thread-locally if we found improvement
          if (best_objective_value_so_far_local < objective_value) {
            objective_state_vector[thread_id].GetCurrentPoint(next_point_local.data());
            best_objective_value_so_far_local = objective_value;
            std::copy(next_point_local.begin(), next_point_local.end(), best_next_point_local.begin());

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
            if (domain.CheckPointInside(best_next_point_local.data()) == false) {
              OL_VERBOSE_PRINTF("WARNING: point outside of domain! point:\n");
              PrintMatrix(best_next_point_local.data(), 1, problem_size);
            }
#endif
          }
        } catch (const std::exception& except) {
          OL_ERROR_PRINTF("Thread %d of %d failed on iteration %d of %d. Message:\n%s\n", thread_id, thread_schedule.max_num_threads, i, num_multistarts, except.what());
          // std::call_once() ensures that the code body here is executed *once* for each unique std::once_flag (we
          // only have 1 instance). Additionally, the operations inside are "atomic" in the sense that no invocation of
          // call_once() will return before the aforementioned single execution is complete (so no risk of partially
          // allocated objects, bad state, etc).

          // Guarantee: only one exception will ever be captured; only one thread will ever execute the lambda function.
          std::call_once(exception_capture_flag, [&captured_exception]() {
              captured_exception = std::current_exception();
            });
        }
      }

#pragma omp critical
      {
        if (io_container->best_objective_value_so_far < best_objective_value_so_far_local) {
          io_container->found_flag = true;
          io_container->best_objective_value_so_far = best_objective_value_so_far_local;
          std::copy(best_next_point_local.begin(), best_next_point_local.end(), io_container->best_point.begin());
        }
      }
    }  // end omp parallel region

    if (unlikely(total_errors != 0)) {
      OL_WARNING_PRINTF("WARNING: %d newton runs exited due to singular Hessian matrices.\n", total_errors);
    }

#ifdef OL_OPTIMIZATION_VERBOSE_PRINT
    if (false == io_container->found_flag) {
      OL_VERBOSE_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
      OL_VERBOSE_PRINTF("Initial guess w/best likelihood:\n");
      PrintMatrix(io_container->best_point.data(), 1, problem_size);
    }
#endif

    if (captured_exception != nullptr) {
      // rethrowing nullptr is illegal
      std::rethrow_exception(captured_exception);
    }
  }

  OL_DISALLOW_COPY_AND_ASSIGN(MultistartOptimizer);
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZATION_HPP_
