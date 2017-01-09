/*!
  \file gpp_exception.hpp
  \rst
  This file contains exception objects along with helper functions and macros
  for exceptions.  This library never calls throw directly. Instead, it provides
  a wrapper in optimal_learning::ThrowException(). Thus we never do::

    throw MyException(...);

  instead preferring one of::

    optimal_learning::ThrowException(MyException(...));  // uncommon
    OL_THROW_EXCEPTION(MyException, ...);  // preferred

  These are analogous to boost::throw_exception() and BOOST_THROW_EXCEPTION.

  ALL exception objects MUST inherit publicly from std::exception. No exceptions (ha ha).

  This file defines the base OptimalLearningException (derived from std::exception) and
  derives several exceptions from it. Each exception type has docs describing the
  output of what() in the class comments.

  Additionally, to use the OL_THROW_EXCEPTION macro (see its #define for details), the
  first two arguments in MyException's ctor must be char *.

  Users may define the macro OL_NO_EXCEPTIONS to *disable* exception
  handling in this library.  Defining that macro means this library will
  never call throw.  Doing so requires users to implement
  optimal_learning::ThrowException() (see comments below for #ifdef OL_NO_EXCEPTIONS).
\endrst*/

#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "gpp_common.hpp"

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXCEPTION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXCEPTION_HPP_

namespace optimal_learning {

/*!\rst
  Macro to stringify the expansion of a macro. For example, say we are on line 53:

  * ``#__LINE__ --> "__LINE__"``
  * ``OL_STRINGIFY_EXPANSION(__LINE__) --> "53"``

  ``OL_STRINGIFY_EXPANSION_INNER`` is not meant to be used directly;
  but we need ``#x`` in a macro for this expansion to work.

  This is a standard trick; see bottom of:
  http://gcc.gnu.org/onlinedocs/cpp/Stringification.html
\endrst*/
#define OL_STRINGIFY_EXPANSION_INNER(x) #x
#define OL_STRINGIFY_EXPANSION(x) OL_STRINGIFY_EXPANSION_INNER(x)

/*!\rst
  Macro to stringify and format the current file and line number. For
  example, if the macro is invoked from line 893 of file gpp_foo.cpp,
  this macro produces the compile-time string-constant:
  ``(gpp_foo.cpp: 893)``
\endrst*/
#define OL_STRINGIFY_FILE_AND_LINE "(" __FILE__ ": " OL_STRINGIFY_EXPANSION(__LINE__) ")"

/*!\rst
  Users may disable exceptions so that this library NEVER invokes throw directly.
  Doing so requires the user to define ThrowException().

  This makes the most sense when paired with the compiler flag -fno-exceptions.
  Using -fno-exceptions will also require disabling Boost's exceptions.
  Ensure the following are defined (preferably by ``#include``) in all files::

    #define BOOST_NO_EXCEPTIONS
    #define BOOST_EXCEPTION_DISABLE

  And provide a definition for::
    void throw_exception( std::exception const & e );

  e.g., the same definition as ThrowException(). See:
  http://www.boost.org/doc/libs/1_55_0/libs/exception/doc/throw_exception.html

  ``throw`` may still be called indirectly through Boost, so disable that
  library's exceptions too. (See below for details/link.)
\endrst*/
#ifdef OL_NO_EXCEPTIONS

/*!\rst
  Disabling exceptions requires the user to implement ThrowException().
  This can be as simple as calling std::abort(). A reference to the
  thrown exception is provided in case other behavior is desired.

  This function normally wraps throw. Callers are allowed to assume
  that this function NEVER returns. If the user-specified implementation
  does, the resulting behavior is UNDEFINED.

  \param
    :exception: an exception object publicly deriving from std::exception
  \return
    **NEVER RETURNS**
\endrst*/
OL_NORETURN void ThrowException(const std::exception& exception);

#else

/*!\rst
  Wrapper around the "throw" keyword, making it easy to disable exceptions.
  Checks that the argument inherits from std::exception and invokes throw.

  \param
    :exception: reference to exception object (publicly deriving from std::exception) to throw
  \return
    **NEVER RETURNS**
\endrst*/
template <typename ExceptionType>
OL_NORETURN inline void ThrowException(const ExceptionType& except) {
  static_assert(std::is_base_of<std::exception, ExceptionType>::value, "ExceptionType must be derived from std::exception.");

  throw except;
}

#endif


/*!\rst
  Macro for throwing exceptions that adds file/line and function name information.
  It is just for convenience, saving callers from having to type OL_STRINGIFY_FILE_AND_LINE,
  and OL_CURRENT_FUNCTION_NAME repeatedly.

  To use this macro, the argument list of ExceptionType's ctor MUST start with two
  ``char const *``, followed by the arguments in ``Args...``.
  Additionally, ExceptionType must be a complete type.

  For example, if you could write::

    throw_exception(BoundsException<double>(OL_STRINGIFY_FILE_AND_LINE, OL_CURRENT_FUNCTION_NAME, "Invalid length scale.", value, min, max));

  then you can instead write::

    OL_THROW_EXCEPTION(BoundsException<double>, "Invalid length scale.", value, min, max);
\endrst*/
#define OL_THROW_EXCEPTION(ExceptionType, Args...) ThrowException(ExceptionType(OL_STRINGIFY_FILE_AND_LINE, OL_CURRENT_FUNCTION_NAME, Args))

/*!\rst
  **Overview**

  Exception to handle general runtime errors (e.g., not fitting into other exception types).
  Subclasses std::exception.
  Serves as the superclass for all other custom exceptions in the ``optimal_learning`` library.

  This class is essentially the same as std::runtime_error but it includes a ctor with
  some extra logic for formatting the error message.

  Holds only a std::string containing the message produced by what().

  .. Note: exceptions from std::string operations (e.g., std::bad_alloc) will cause std::terminate().

  **Message Format**

  The ``what()`` message is formatted in the class ctor (capitals indicate variable information)::

    R"%%(
    OptimalLearningException: CUSTOM_MESSAGE FUNCTION_NAME FILE_LINE_INFO
    )%%"

  This format should be overriden by subclasses (at the minimum showing a different exception name).
\endrst*/
class OptimalLearningException : public std::exception {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "OptimalLearningException";

  /*!\rst
    Constructs a OptimalLearningException with the specified message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
  \endrst*/
  OptimalLearningException(char const * line_info, char const * func_info, char const * custom_message);

  /*!\rst
    Provides a C-string containing information about the conditions of the exception.
    See: http://en.cppreference.com/w/cpp/error/exception

    \return
      C-style char string describing the exception.
  \endrst*/
  virtual const char* what() const noexcept override OL_WARN_UNUSED_RESULT {
    return message_.c_str();
  }

  OptimalLearningException() = delete;

 protected:
  /*!\rst
    Constructs a OptimalLearningException with the specified name.
    This is used by subclasses to override the class name in the message text.

    \param
      :name[]: the exception name to write into the message
  \endrst*/
  explicit OptimalLearningException(char const * name);

  /*!\rst
    Utility function to append some additional info (file/line number, function name,
    and/or a custom message) to a specified string.
    This is meant to be used for constructing what() messages for the exception classes
    in gpp_exception.hpp.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
\endrst*/
  void AppendCustomMessageAndDebugInfo(char const * line_info, char const * func_info,
                                       char const * custom_message);

  //! a custom message describing this exception, produced by ``what()``.
  std::string message_;
};

/*!\rst
  **Overview**

  Exception to capture value < min_value OR value > max_value.

  Stores value, min, and max for debugging/logging/reacting purposes.

  **Message Format**

  The ``what()`` message is formatted in the class ctor (capitals indicate variable information)::

    R"%%(
    BoundsException: VALUE is not in range [MIN, MAX].
    CUSTOM_MESSAGE FUNCTION_NAME FILE_LINE_INFO
    )%%"
\endrst*/
template <typename ValueType>
class BoundsException : public OptimalLearningException {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "BoundsException";

  /*!\rst
    Constructs a BoundsException object with extra fields to flesh out the what() message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :value: the value that violates its min or max bound
      :min: the minimum bound for value
      :max: the maximum bound for value
  \endrst*/
  BoundsException(char const * line_info, char const * func_info,
                  char const * custom_message, ValueType value_in,
                  ValueType min_in, ValueType max_in);

  ValueType value() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return value_;
  }

  ValueType max() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return max_;
  }

  ValueType min() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return min_;
  }

  OL_DISALLOW_DEFAULT_AND_ASSIGN(BoundsException);

 protected:
  BoundsException(char const * name_in, char const * line_info,
                  char const * func_info, char const * custom_message,
                  ValueType value_in, ValueType min_in, ValueType max_in);

 private:
  //! The errorneous value_ and the ``[min_, max_]`` bounds that it should lie in.
  ValueType value_, min_, max_;
};

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template class BoundsException<int>;
extern template class BoundsException<double>;

/*!\rst
  Exception to capture value < min_value.
  Simple subclass of BoundsException that sets the max argument to std::numeric_limits<ValueType>::max()

  See BoundsException for ``what()`` message format.
\endrst*/
template <typename ValueType>
class LowerBoundException : public BoundsException<ValueType> {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "LowerBoundException";

  /*!\rst
    Constructs a LowerBoundException object with extra fields to flesh out the what() message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :value: the value that violates its min or max bound
      :min: the minimum bound for value
  \endrst*/
  LowerBoundException(char const * line_info, char const * func_info,
                      char const * custom_message, ValueType value_in,
                      ValueType min_in)
      : BoundsException<ValueType>(kName, line_info, func_info, custom_message, value_in,
                                   min_in, std::numeric_limits<ValueType>::max()) {
  }

  OL_DISALLOW_DEFAULT_AND_ASSIGN(LowerBoundException);
};

/*!\rst
  Exception to capture value > max_value.
  Simple subclass of BoundsException that sets the min argument to std::numeric_limits<ValueType>::lowest()

  See BoundsException for ``what()`` message format.
\endrst*/
template <typename ValueType>
class UpperBoundException : public BoundsException<ValueType> {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "UpperBoundException";

  /*!\rst
    Constructs an UpperBoundException object with extra fields to flesh out the what() message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :value: the value that violates its min or max bound
      :max: the maximum bound for value
  \endrst*/
  UpperBoundException(char const * line_info, char const * func_info,
                      char const * custom_message, ValueType value_in,
                      ValueType max_in)
      : BoundsException<ValueType>(kName, line_info, func_info, custom_message, value_in,
                                   std::numeric_limits<ValueType>::lowest(), max_in) {
  }

  OL_DISALLOW_DEFAULT_AND_ASSIGN(UpperBoundException);
};

/*!\rst
  **Overview**

  Exception to capture value != truth (+/- tolerance).
  The tolerance parameter is optional and only usable with floating point data types.

  Stores value and truth (and tolerance as applicable) for debugging/logging/reacting purposes.

  **Message Format**

  The ``what()`` message is formatted in the class ctor (capitals indicate variable information)::

    R"%%(
    InvalidValueException: VALUE != TRUTH (value != truth).
    CUSTOM_MESSAGE FUNCTION_NAME FILE_LINE_INFO
    )%%"

  OR ::

    R"%%(
    InvalidValueException: VALUE != TRUTH ± TOLERANCE (value != truth ± tolerance).
    CUSTOM_MESSAGE FUNCTION_NAME FILE_LINE_INFO
    )%%"

  Depending on which ctor was used.
\endrst*/
template <typename ValueType>
class InvalidValueException : public OptimalLearningException {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "InvalidValueException";

  /*!\rst
    Constructs a InvalidValueException object with extra fields to flesh out the what() message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :value: the invalid value
      :truth: what "value" is supposed to be
  \endrst*/
  InvalidValueException(char const * line_info, char const * func_info,
                        char const * custom_message, ValueType value_in, ValueType truth_in);

  /*!\rst
    Constructs a InvalidValueException object with extra fields to flesh out the what() message.
    This ctor additionally has an input for tolerance, and is only enabled for floating point types.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :value: the invalid value
      :truth: what "value" is supposed to be
      :tolerance: the maximum acceptable error in ``|value - truth|``
  \endrst*/
  template <typename ValueTypeIn = ValueType, class = typename std::enable_if<std::is_floating_point<ValueType>::value, ValueTypeIn>::type>
  InvalidValueException(char const * line_info, char const * func_info,
                        char const * custom_message, ValueType value_in,
                        ValueType truth_in, ValueType tolerance_in);

  ValueType value() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return value_;
  }

  ValueType truth() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return truth_;
  }

  ValueType tolerance() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return tolerance_;
  }

  OL_DISALLOW_DEFAULT_AND_ASSIGN(InvalidValueException);

 private:
  //! the erroneous ``value_`` and the ``truth_ +/- tolerance_`` range that it should lie in
  ValueType value_, truth_, tolerance_;
};

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
extern template class InvalidValueException<int>;
extern template class InvalidValueException<double>;
extern template InvalidValueException<double>::InvalidValueException(
    char const * line_info, char const * func_info, char const * custom_message,
    double value_in, double truth_in, double tolerance_in);

/*!\rst
  **Overview**

  Exception to capture when a *square* matrix ``A`` (``\in R^{m x m}``) is singular.

  Stores the matrix (in a std::vector) and its dimensions along with the index of the leading minor that is non-SPD.

  .. Note:: std::vector<double> ctor can throw and cause std::terminate().

  **Message Format**

  The ``what()`` message is formatted in the class ctor (capitals indicate variable information)::

    R"%%(
    SingularMatrixException: M x M matrix is singular; i-th leading minor is not SPD.
    CUSTOM_MESSAGE FUNCTION_NAME FILE_LINE_INFO
    )%%"

  .. Note:: this exception currently does not print the full matrix. Use a debugger
    and call PrintMatrix() (gpp_logging.hpp) or catch the exception and
    proecss the matrix.
\endrst*/
class SingularMatrixException : public OptimalLearningException {
 public:
  //! String name of this exception for logging.
  constexpr static char const * kName = "SingularMatrixException";

  /*!\rst
    Constructs a SingularMatrixException object with extra fields to flesh out the what() message.

    \param
      :line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
      :func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
      :custom_message[]: optional ptr to char array with any additional text/info to print/log
      :matrix[num_rows][num_cols]: the singular matrix
      :num_rows: number of rows (= number of columns) in the matrix
      :leading_minor_index: index of the first non-positive definite (principal) leading minor
  \endrst*/
  SingularMatrixException(char const * line_info, char const * func_info, char const * custom_message,
                          double const * matrix_in, int num_rows_in, int leading_minor_index_in);

  int num_rows() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_rows_;
  }

  int leading_minor_index() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return leading_minor_index_;
  }

  const std::vector<double>& matrix() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return matrix_;
  }

  OL_DISALLOW_DEFAULT_AND_ASSIGN(SingularMatrixException);

 private:
  //! the number of rows (= number of columns) in the singular matrix
  int num_rows_;
  //! index of the first non-positive definite (principal) leading minor
  int leading_minor_index_;
  //! the data of the singular matrix, ordered column-major
  std::vector<double> matrix_;
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXCEPTION_HPP_
