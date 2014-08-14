/*!
  \file gpp_common.hpp
  \rst
  This file contains some additional style specifications and notes on implementation details in the header
  comments.

  This file also contains a few compiler macros that are used in most files throughout this project.
  This includes macros for disallowing constructors, common constants, gcc compiler hints, etc.  The usage
  and meaning of these are detailed in comments above their definitions.

  I am aware that the Google Style Guide eschews compiler macros but I don't see a better way to do most
  of this.

  **IMPLEMENTATION NOTES**

  1. No function in this project allocates memory that the caller must free.
     In particular, this means that array outputs must be allocated by the caller.
     Additionally, class members should never be plain pointers--use std::unique_ptr to guarantee that
     memory is released appropriately.

     In fact, ``new`` and ``delete`` (as they pertain to memory) should almost NEVER be used.

     Exception: ``Clone()`` functions and inputs to std::unique_ptr's ctor may use ``new``.

     Finally, ``malloc``/``free`` should be AVOIDED ENTIRELY.  Use std::vector instead of malloc'd arrays and
     std::unique_ptr when singleton object allocation is desired.  If you *ABSOLUTELY* must, use ``new``/``delete``
     and ``new[]``/``delete[]``, NOT ``malloc``/``free``.

  2. Matrices are stored as flattened arrays and indexed in COLUMN-MAJOR order.
     2D (and more generally nD) matrices are strictly NOT ALLOWED.

     However, for convenience, C multi-dimensional indexing will be used to describe
     matrices: e.g., A[dim1][dim2] is a matrix with dim1 rows and dim2 columns.  But
     dim1 is the most rapidly-varying index here.
     for example::

       A[3][4] =
         [4  32  5  2
          53 12  8  1
          81  2  93 0]

      would be FLATTENED into an array:

      ``A_flat[12] = [4 53 81 32 12 2 5 8 93 2 1 0]``

      So ``A[dim_1][dim_2]...[dim_n]`` would be represented as an array of size

      ``dim_1*dim_2*...*dim_n``.

      If I wanted to access ``A[i_1][i_2]...[i_n]`` in a multi-dimensional array, I would perform the following
      in a flat array:

      ``A[i_n*dim_1*dim_2*...*dim_{n-1} + ... + i_2*dim_1 + i_1]`` (note that ``dim_n`` never appears in the index!)

      (See gpp_math.cpp header comments for more details on a more efficient way of array accessing.)

   3. The meaning of ``lda, ldb, ldc``.  ``lda``, for example, is short for
      "leading dimension [of the matrix] A".  This is notion is equivalent to that of column-stride,
      the number of elements between adjacent elements of a row.  (Note we always use unit row-stride
      for matrix storage; i.e., adjacent elements in a column are stored next to each other in memory.)
      These parameters are useful for indicating that a matrix is allocated larger than the specified size.
      For example, if we wanted to compute a matrix product using only the leading 5x5 submatrix of A and
      A is allocated as ``A[25][25]``, we could call multply with the size set to 5 and
      lda set to 25.  This avoids needless copying while maintaining reasonable locality, since
      the row-stride remains 1 and the column-stride is 25.  Compare to an array allocated as ``B[5][5]``,
      which has row-stride 1 and column-stride 5.

      Additionally, note that this technique is more general than just obtaining leading principal submatrices; we
      can obtain any submatrix.  For example, to get ``A[2:6][8:15]`` from ``A[25][25]``, we would pass in ``A' = A + 8*25 + 2``, which
      points to the first element of the desired submatrix.  Then we set ``n_rows = 5, n_columns = 8, lda = 25``.

   4. Keywords:
      In general, see details in the body of this file (e.g., if you don't know what "likely" is, search for
      "#define likely") for the various specifiers and keywords we use heavily.  gcc documentation, stackoverflow,
      cppreference, and wikipedia can be additional good sources of info.

      a. mark all pointers ``const`` whenever appropriate (e.g., all inputs)
      b. mark all class member functions ``const`` whenever appropriate (this should be nearly all member functions!)
      c. mark all pointers ``restrict`` whenever appropriate (see details in the definition of restrict below for more info)
         in fact, essentially ALL pointers should be marked ``restrict``
      d. we use some gcc compiler hints: ``__builtin_expect`` to express ``likely`` and ``unlikely`` as hints about when
         loop branches are very common or uncommon (do not be overzealous with this!)
      e. we also use several gcc attributes:

         * ``unused``: suppresses compiler warnings when we expect variables/functions to be unsused
         * ``nonnull_pointers``: compiler warnings when pointer args are ``nullptr`` (see details for special case with class member fcns)
         * ``warn_unused_result``: compiler warning when return value is unused
         * ``__pure__, __const__``: describes a function's dependencies/effects on global state
         * ``final``: no subclass can derive from this (class level) or no function can override this (fcn level)
         * ``override``: this function MUST override a base class method
         * ``noexcept``: this function CANNOT throw exceptions

   5. When designing objects, separate them from their [mutable] state.  This is to make it possible to explicitly see
      which components should be constant during execution and which components are stateful or temporary.  This separation
      (or something like it) is also necessary for avoiding repeated work/allocation during multithreading by clearly
      labeling what data must be thread-local.

      The following discussion draws its examples from the objects in gpp_math.hpp/cpp.
      For example, if the "core" object is LogMarginalLikelihoodEvaluator, then its state might be LogMarginalLikelihoodState.
      For convenience, the "core" object should register its State type in a type alias for "StateType"; e.g.,
      ``  using StateType = FooState;``

      And the state object should register an "EvaluatorType" (or whatever name is convenient); e.g.,
      ``  using EvaluatorType = FooEvaluator;``

      State objects will be structs, possibly with simple routines to help with memory management/setup.  They should have
      no complex logic, as manipulating their data should only be done by the "core" object.

      Foo's state should never be manipulated directly.  Instead, Foo should initialize its State, and then its State
      should be passed as a parameter to its member functions.  This relationship is explained in detail in the docs for
      GaussianProcess and its state, PointsToSampleState.  In general, the setup will look like::

        struct FooState;  // forward declaration needed for enums, pointers, references in Foo
        class Foo {
         public:
          using StateType = FooState;

          FillState(StateType *) const {  <-- usually not called directly, use FooState.SetupState instead
            ...direct manipulation of FooState members...
          }

          ComputeSomethingThatUsesState(const StateType&) const {
            ...
          }

          ComputeSomethingThatModifiesState(StateType *) const {
            ...
          }
        }

        struct FooState {
          FooState(const Foo& foo) {
            SetupState(foo);
          }

          SetupState(const Foo& foo) { <-- safest way to setup state; ensures data sizes are correct
            ...code to ensure data size is correct...
            foo.FillState(this);  <-- call foo to configure self
          }

          ...public data members...
        }

      Then these would be used::

        ComputeSomethingWithFoo(const Foo& foo) {
          Foo::StateType foo_state(foo);
          foo.ComputeSomething(foo_state);
        }

      Or::

        ComputeSomethingWithFoo(const Foo& foo, Foo::StateType * foo_state) {
          foo_state.SetupState(foo);  <-- optional depending on preconditions
          foo.ComputeSomething(foo_state);
        }

      Finally, the data members of a State are usually broken into two parts:

      a. Problem specification:
         e.g., size data, hyperparameters, points being sampled or other quantities
         that would be updated over the course of say gradient descent
      b. Derived quantities:
         These are members that are a function of the problem specification that are also
         worth precomputing.  This is what
         ``foo_state.SetupState(foo);``
         would initialize, where applicable.
      c. Temporary storage:
         These members are more like pre-allocated space for work in State's consumers; e.g., for
         storing intermediate results.  Temporary storage can be overwitten with anything and have unreliable state.
         The State's constructor must size them properly, but their contents can be *anything*.
         DO NOT rely on the values in these members!  Set before using!

      =================================================  ===============================================
      List of classes implementing the Evaluator/State relationship
      --------------------------------------------------------------------------------------------------
      Class                                               State
      =================================================  ===============================================
      GaussianProcess                                     PointsToSampleState
      ExpectedImprovementEvaluator                        ExpectedImprovementState
      OnePotentialSampleExpectedImprovementEvaluator      OnePotentialSampleExpectedImprovementState
      LogMarginalLikelihoodEvaluator                      LogMarginalLikelihoodState
      LeaveOneOutLogLikelihoodEvaluator                   LeaveOneOutLogLikelihoodState
      =================================================  ===============================================

      One set of noteable exceptions to this 'rule' is the RNG classes.  These objects' sole purpose
      is to hold mutable state so there's nothing to be gained from just splitting code and data members.

   6. Explicit [template] Instantiation.

      Summarizing the syntax briefly:

      **FOR FUNCTIONS**

      IN THE HPP FILE::

        template <typename type1, typename type2, ...>
        void SomeFunction(int arg1, type1 arg2, ...);  // template function declaration

        extern template void SomeFunction(int, SPECIFIC_TYPE arg2, ...);  // explicit instantiation DECLARATION
        // you must specify all template parameters (type1, type2, ...) and replace them in the fcn's argument list

      END HPP FILE

      IN THE CPP FILE::

        template <typename type1, typename type2, ...>
        void SomeFunction(int arg1, type1 arg2, ...) {  // template function definition
          ...your code here...
        }

        template void SomeFunction(int, SPECIFIC_TYPE arg2, ...);  // explicit instantiation DEFINITION
        // this is identical to explicit instantiation line in the HPP file, except no "extern" keyword

      END CPP FILE


      **FOR CLASSES**

      IN THE HPP FILE::

        template <typename type1, typename type2, ...>
        class SomeClass {  // template class declaration
          ...
        };

        extern template class SomeClass<SPECIFIC_TYPE>;  // extern declaration telling compiler NOT to instantiate the
                                                         // current translation unit

      END HPP FILE


      IN THE CPP FILE::

        SomeClass::SomeMethod(...) {  // any implementations you are hiding
          ...
        }

        template class SomeClass<SPECIFIC_TYPE>;  // obliges compiler to instantiate a template

      END CPP FILE

      .. Note:: this is NOT the same as template SPECIALIZATION.

      **What happens when the compiler sees a templated call?**

      When the C++ compiler encounters a templated call, it will instantiate the template with the appropriate types and compile
      that code.  This is done once per type combination, per compilation unit.  To do this, the definition of the template
      must be visible.  At link-time, the (potentially) duplicate copies are coalesced.

      **What is explicit instantiation?**

      The explicit instantiation definition tells the compiler to emit code for that particular combination of template
      parameters in the present compilation unit.

      The explicit instantiation declaration tells the compiler that a symbol for the instantiated call (with that specific
      combination of template parameters) exists somewhere.  (This is identical to the usage of "extern" in C.)  It will leave
      an "endpoint" that the linker will connect later.

      **Why use explicit instantiation?**

      a. Only way to expose templated functionality in libraries (e.g., .dll/.so/.a) WITHOUT exposing the implementation.

         So if you do not want to expose the template definition in a header, explicit instantiation is the only way to
         provide functionality from templates in libraries.

      b. Speeds up compilation.

         If you have the same type combinations in many compilation units, this can slow compile time substantially, especially
         for complex templates.

      c. Restricts how your template can be used (e.g., if you are using templates more like glorified C macros).

         No instantiations (in compilation units outside the unit containing the definition) are possible except the
         explicit ones.

      In this project, B and C are our main reasons for template instantiation.  It keeps compile-times down and we are mostly
      using templates to save copy-pasting code (let the compiler do it for you through templates!).  Think of it as static
      polymorphism.  Generally we explicitly instantiate large/complex templates and leave simple definitions in the header
      (also keeps headers clean).

      For those unfamiliar with the concept, further reading:
      http://pic.dhe.ibm.com/infocenter/lnxpcomp/v111v131/index.jsp?topic=%2Fcom.ibm.xlcpp111.linux.doc%2Flanguage_ref%2Fexplicit_instantiation.html
      http://msdn.microsoft.com/en-us/library/7k8twfx7.aspx
      http://stackoverflow.com/questions/2351148/explicit-instantiation-when-is-it-used

   7. Matrix-loop Idiom
      The matrix looping idioms used in this file deserve some explanation; we'll use matrix-vector-multiply
      as an example.  One common implementation of ``y = A * x``, with a ``m x n``, row-major matrix is::

        double const * restrict A = ...;
        double const * restrict x = ...;
        double * restrict y = ...;
        for (int i = 0; i < m; ++i) {
          y[i] = 0;
          for (int j = 0; j < n; ++j) {
            y[i] += A[i*n + j]*x[j]
          }
        }

      Instead, we use::

        for (int i = 0; i < m; ++i) {
          y[i] = 0;
          for (int j = 0; j < n; ++j) {
            y[i] += A[j]*x[j];
          }
          A += n;
        }

      Pros:

      a. Avoids extra integer arithmetic: redundant ``i*n`` computations removed. (Compiler may save off ``i*n``)
         Also greatly reduces number of characters: indexing ``A[i*dim_1*dim_2 + j*dim_2 + k]`` for example is unwieldy.
      b. Simple offsets like ``A[j]`` can be computed with just one x86 instruction, ``LEA``.  (``A[i*n + j]`` requires 2-3)
         ``LEA`` cannot handle ``A[g + j]``, so compilers (currently) will not optimize this for you.
      c. This style makes it more clear that the inner loop is just a dot-product.  This may also improve the compiler's
         ability to vectorize.

      Cons:

      a. Overwrites the pointer ``A``.  If the full ``A`` matrix is required again, we need to store it before this loop
         or set ``A -= m*n``; after the loop.  (Compiler optimization makes this a non-issue.)
      b. Following a., leaves the pointer ``A`` in an invalid state.  After the loop completes, ``A[0]`` would generally
         dereference unallocated space.
      c. Is not the most common idiom and may feel unfamiliar to some programmers.

   8. RAII (Resource Acquisition Is Initialization)
      We use RAII, no exceptions.
      RAII ensures (amongst other things) that allocation/deallocation are always tightly coupled (through dtors): when
      an object goes out of scope, its resources are released. So it is impossible to forget to call delete after new and
      no additional code is needed to ensure delete is called if an exception is thrown.

      In particular, you will not make "bare" calls to new/delete (or malloc/free), fopen/fclose, etc. nor will you
      implement 'bare' locks in ints, etc. Use the proper object or container (even if it's just std::unique_ptr or similar).

      For more information, see:
      http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization
      http://www.stroustrup.com/except.pdf

   9. Exception Safety
      All library components provide "basic" exception safety. "Basic" safety means that all invariants of the component
      are preserved and no resources are leaked. See references below for more details/definitions. RAII (see item 9)
      makes providing the basic guarantee all but trivial.

      Some components provide stronger guarantees (e.g., functions marked noexcept).
      Additionally, operations on Evaluator/State pairs (see item 5) can be endowed with "strong" exception safety
      easily if desired. Simply provide a wrapper class that saves the state and resets if an exception is thrown.
      For example::

        struct State {
          void ComputationWithBasicGuarantee(...);
          void SetCurrentPoint(double *);
          void GetCurrentPoint(double *);
          int GetProblemSize();
        };

        struct StateWrapper : public State {
          void ComputationWithStrongGuarantee(...) {
            std::vector<double> point(GetProblemSize());
            GetCurrentPoint(point.data());  // save valid, original state
            try {
              ComputationWithBasicGuarantee(...);
            } catch (const std::exception& e) {
              SetCurrentPoint(point.data());  // reset state
              throw;
            }
          }
        };

      .. Note:: if State has a random number generator, then its internal state may need to be saved as well. Generally this
        is NOT necessary! The RNG is guaranteed to be in a valid state, and users should have no reason to care what
        the precise state of the RNG is. (Still if you do care, the RNG wrappers in gpp_random.hpp can save state.)

      See here for further information on the different types of exception safety:
      http://en.wikipedia.org/wiki/Exception_safety
      http://www.boost.org/community/exception_safety.html

  Lastly, some general mathematical notation:
  A vector "x" of length size may be represented: (LOWER CASE letters)

  ``x[size], x_i``

  A matrix "A" of size size_1 x size_2 (#rows x #columns) may be represented: (UPPER CASE letters)

  ``A[size_1][size_2], A_{i,j}, A_ij``

  ``A_i`` may also be used to refer to the i-th column of a matrix A.

  Matrices are stored flat in column-major (implementation note 2).  So the concept ``A[i][j]`` is actually accessed via:

  ``A[j*size_1 + i]``

  Finally, a note about function comment style:

  a. comments go above the function declaration or definition they describe
  b. declaration comments should individually describe all inputs/ouputs/returns using RST markdown; e.g., ::

       BEGIN_COMMENT!\rst
       Compute all the stuff.

       \param
         :size: number of variables
         :x[size]: vector of input variables
         :A[size][size]: matrix describing relation of i-th, j-th input variables
       \output
         :y[size]: vector of computed results
       \return
         returns the confidence score of the results, y
       \endrstEND_COMMENT
       double ComputeStuff(int size, double const * restrict x, double const * restrict A, double * restrict y);

     Here, ``A[size][size]`` indicates the size and data-ordering of ``A``: it has size rows and size columns, and is stored
     column-major (see implementation note 2).

     .. NOTE:: if we have a ``char*`` argument, ``my_str``, we will generally not specify the
       array size, instead writing:

       ``my_str[]: a pointer to a char array``

       since std::strlen() can be used to find the length.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_COMMON_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_COMMON_HPP_

namespace optimal_learning {

/*!\rst
  Macros declare copy and assignment operators for C++ classes.  It is intended to place these
  in the private (or possibly protected) segments of a class to disallow these actions.

  Unless absolutely necessary, classes should disallow both of these operations.
\endrst*/
#define OL_DISALLOW_DEFAULT_CONSTRUCTOR(TypeName) \
  TypeName() = delete

#define OL_DISALLOW_COPY(TypeName) \
  TypeName(const TypeName&) = delete

#define OL_DISALLOW_ASSIGN(TypeName) \
  void operator=(const TypeName&) = delete

#define OL_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  OL_DISALLOW_COPY(TypeName);                 \
  OL_DISALLOW_ASSIGN(TypeName)

#define OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(TypeName) \
  OL_DISALLOW_DEFAULT_CONSTRUCTOR(TypeName); \
  OL_DISALLOW_COPY_AND_ASSIGN(TypeName)

#define OL_DISALLOW_DEFAULT_AND_ASSIGN(TypeName) \
  OL_DISALLOW_DEFAULT_CONSTRUCTOR(TypeName); \
  OL_DISALLOW_ASSIGN(TypeName)

/*!\rst
  Macros for printing the name of the function immediately containing the macro.
  __PRETTY_FUNCTION__ is preferred when supported; it prints additional information about
  template parameters and function arguments.

  Note that currently in C++ (gcc 4.x.x), both __func__ and __PRETTY_FUNCTION__ are
  variables, NOT macros! The compiler implicitly declares:
  static const char __func__[] = "some_name";
  http://stackoverflow.com/questions/4384765/whats-the-difference-between-pretty-function-function-func

  Redefine to empty-string if you do not want this behavior.
\endrst*/
#ifdef __GNUC__
#define OL_CURRENT_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define OL_CURRENT_FUNCTION_NAME __func__
#endif

/*!\rst
  Macros that call compiler hints to aid in branch prediction.  These use ``__builtin_expect()``,
  which is a gcc extension to the C language:
  http://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/Other-Builtins.html
  It's at least supported by gcc and icc.

  Examples::

    if (likely(condition)) {
      DoTask();
    } else {
      HandleError();
    }

  Don't be overzealous with this hint: it's the most applicable for:

  1. branches that almost never happen (e.g., errors)
  2. branches that need to run as fast as possible

  ``__builtin_expect()`` can also be used with switch-statements::

    switch (__builtin_expect(switch_value, most_common_value)) {
      case 1: {
      }
      case 2: {
      }
      case most_common_value: {
      }
      case 3: {
      }
      ...
      default: {
      }
    }

  Although here it is probably better to fold out most_common_value with an ``if``-statement and
  put the ``switch`` in the ``else`` clause.
\endrst*/
#ifndef unlikely
#define unlikely(expr) __builtin_expect((expr), 0)
#endif

#ifndef likely
#define likely(expr) __builtin_expect((expr), 1)
#endif

/*!\rst
  Macro to allow ``restrict`` as a keyword for ``C++`` compilation.
  ``restrict`` was a keyword added to the ``C`` programming language as part of the ``C99`` standard; it boosts ``C`` performance to the
  level of Fortran as the issue ``restrict`` addresses (pointer aliasing) was one of the biggest differences between the two.

  ``restrict`` is *NOT* a part of any ``C++`` standard (not even ``C++11``).  However, due to its extreme
  usefulness, most C++ compilers (e.g., gcc, icc) support it.

  See references below for full details.  In short, the ``restrict`` keyword promises that ``restrict``'d pointers will not
  *WRITE ALIAS* or *READ/WRITE ALIAS* each other.  This extends to quantities deived from a ``restrict``'d pointer, e.g., ::

    double * restrict x;

  Then no other pointer may alias ``x, x + 1, x + 2``, etc.

  Declare like::

    some_type * restrict pointer_to_some_type;

  Note that since C/C++ declarations are read read right-to-left (e.g., pointer_to_some_type is a ``restrict``'d pointer to
  some_type), the following makes no sense::

    some_type restrict * illogical_pointer;

  For multi-dimensional pointers, typically only::

    double ** restrict matrix;

  is needed, but sometimes::

    double * restrict * restrict matrix;

  or::

    double * restrict * matrix;

  are appropriate.

  Suppose I have::

    double * restrict a = ...;
    double * restrict b = ...;

  Here, I promise that values WRITTEN through a will NEVER affect values read through b and vice versa.  In particular,
  pointers that are only used for reading may be marked ``restrict`` and still alias each other with no bad effects.  If you
  need ``restrict``'d pointers for only a sub-part of a function, drop a scope { }.

  For example, consider ``y = A*x``::

    double matrix_vector_multiply(double * A, double *x, int size, double * y) {
      for (int i=0; i<size; ++i) {
        y[i] = 0.0;
        for (int j=0; j<size; ++j) {
          y[i] += A[i*size +j] * x[j];
        }
      }
    }

  The compiler has to emit code that would be correct for::

    matrix_vector_multiply(A, A, size, A);

  So A is being read/written simultaneously.  In practical use, this makes no sense whatsoever.  But the compiler has to guarantee
  that this case would be executed correctly, which severely limits optimizations since the aliasing possibility forces full
  serialization of the entire loop.

  Instead::

    double matrix_vector_multiply(double const * restrict A, double const * restict x, int size, double * restrict y);

  tells the compiler that ``A, x``, wil never read from values written through ``y``.  (The same is true for ``y`` never reading from values
  written from ``A, x``; but this is trivial since ``A, x`` are now marked as pointers to const double.)

  .. NOTE:: the compiler DOES NOT GUARANTEE that ``restrict`` is used correctly.  Invalid use of ``restrict`` will lead to undefined behavior.

  You should essentially be able to mark EVERY pointer as ``restrict``.  The aliased behavior allowed by not using ``restrict`` is
  exceedingly confusing and SLOW, and it should be avoided.
  Mark const whenever appropriate.

  References:
  http://en.wikipedia.org/wiki/Restrict
  http://publib.boulder.ibm.com/infocenter/comphelp/v7v91/index.jsp?topic=%2Fcom.ibm.vacpp7a.doc%2Flanguage%2Fref%2Fclrc03restrict_type_qualifier.htm
  http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html
\endrst*/
#ifdef __cplusplus
#define restrict __restrict__
#endif

/*!\rst
  gcc macro to label functions & function parameters as unused.  This suppresses a large number of warnings.
  We may have unused parameters in templated functions--say a template written to handle matrices but also used
  for vectors (treated as a ``1 x n`` matrix; then the leading dimension is unused)

  Recall that ``##`` concatenates, so that the name ``foo`` in ``OL_UNUSED(foo)`` is expanded to ``UNUSED_foo``
\endrst*/
#ifdef __GNUC__
#define OL_UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
#define OL_UNUSED(x) UNUSED_ ## x
#endif

/*!\rst
  gcc macro to specify (= promise the compiler) that no ``nullptr`` pointers will be passed as function args.
  Warnings will be issued if the compiler can determine the rule is being violated.  Runtime checks are NOT
  performed, so "cheating" and passing null pointers results in undefined behavior.

  Note about the use of ``OL_NONNULL_POINTERS_LIST``: Indexing starts from 1 at left-most function arg for ``C`` functions
  and non-class member ``C++`` functions; from 2 otherwise.  In ``C++`` member functions, the first argument is implicitly
  the ``this`` pointer, hence why indexing starts at 2.
  DO NOT specify ``this`` in the list in ``C++`` member functions!  It is undefined behavior for ``this`` to be ``nullptr`` so
  this attribute makes no sense for ``this``.
\endrst*/
#ifdef __GNUC__
#define OL_NONNULL_POINTERS __attribute__((__nonnull__))  // all pointers must be nonnull
#define OL_NONNULL_POINTERS_LIST(...) __attribute__((__nonnull__ (__VA_ARGS__)))  // pointers in specified positions must be
// nonnull.
#else
#define OL_NONNULL_POINTERS
#define OL_NONNULL_POINTERS_LIST(...)
#endif

/*!\rst
  gcc macro to enable warning if the return value of a function is not used.
\endrst*/
#ifdef __GNUC__
#define OL_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define OL_WARN_UNUSED_RESULT
#endif

/*!\rst
  gcc macro to label functions as "const".  Such a function has NO SIDE EFFFECTS and no effects beyond the return value.
  Additionally, a pure function's return depends only on input parameters--NO global memory.  In particular, a const
  function cannot dereference pointers.
\endrst*/
#ifdef __GNUC__
#define OL_CONST_FUNCTION __attribute__((__const__))
#else
#define OL_CONST_FUNCTION
#endif

/*!\rst
  gcc macro to label functions as "pure".  Pure functions are like const functions except they are additionally
  allowed to READ global memory (but not write--no side effects!).  See gcc docs for more information.
\endrst*/
#ifdef __GNUC__
#define OL_PURE_FUNCTION __attribute__((__pure__))
#else
#define OL_PURE_FUNCTION
#endif

/*!\rst
  Some functions never return, e.g., ``abort()``, ``terminate()``, or functions that
  always invoke throw.  noreturn allows gcc to generate cleaner code; it also
  makes the required behavior clear.
\endrst*/
#ifdef __GNUC__
#define OL_NORETURN __attribute__((__noreturn__))
#else
#define OL_NORETURN
#endif

/*!\rst
  icc ``C++11`` support before v14.x.x is incomplete and does not support the ``override`
  or ``final`` specifiers (although "final" appears to work). See:
  https://software.intel.com/en-us/articles/c0x-features-supported-by-intel-c-compiler
\endrst*/
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1400)
#define override
#endif

/*!\rst
  Square a ``constexpr`` value.
  For details on what ``constexpr`` expects:
  http://www.stroustrup.com/C++11FAQ.html#constexpr
  http://en.cppreference.com/w/cpp/language/constexpr

  Basically, the compiler *can* evaluate this function at compile time, so its structure/inputs are restricted
  to things that could potentially be compile-time constants.

  This means SQ will fail with non-constexpr inputs.

  .. WARNING:: this is pass-by-VALUE.  In combination with constexpr, this fcn is only meant for very simple operations.
    (i.e., this shouldn't be used with a matrix class supporting operator*, and it probably wouldn't compile anyway.)
    If you need pass-by-reference, be warned that it ``Square(const T&)`` may fail in icc:

    http://software.intel.com/en-us/articles/c0x-features-supported-by-intel-c-compiler
    http://software.intel.com/en-us/forums/topic/391885

    They claim it'll be available in the 14.0 compiler.

  \param
    :value: value to be squared
  \return
    the product: value * value
\endrst*/
template <typename T>
constexpr OL_WARN_UNUSED_RESULT OL_CONST_FUNCTION T Square(T value) {
  return value*value;
}

/*!\rst
  Work-around for compilers (icc <= v13.0, gcc <= v4.5, msvcc <= 2013) that cannot deal with::

    struct Foo {
      SomeType i;
    };
    Foo f;
    SomeType y = decltype(f)::i;

  That is, resolve the scoping operator with ``decltype()``.  Instead, do::

    SomeType y = IdentifyType<decltype(f)>::type::i;

  This can be removed once the relevant compilers support ``decltype`` + scoping.
\endrst*/
template <typename T>
struct IdentifyType {
  using type = T;
};

/*!\rst
  Macros for a few useful mathematical constants.
\endrst*/
static constexpr double kPi = 3.1415926535897932384626;
static constexpr double kSqrt3 = 1.7320508075688772935274;
static constexpr double kSqrt5 = 2.2360679774997896964092;
static constexpr double kLog2Pi = 1.8378770664093454835607;

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_COMMON_HPP_
