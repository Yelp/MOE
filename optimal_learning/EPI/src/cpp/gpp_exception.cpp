// gpp_exception.cpp
/*
  This file contains definitions for the constructors of the various exception classes in gpp_exception.hpp. These ctors
  generally set the message_ member with some debugging information about what the error is and where it occurred.

  In most cases, we use boost::lexical_cast<std::string> to convert from numbers to strings. std::to_string's formatting
  for floating point types is absolutely terrible (but it works fine for integral types, which is where we use it).
*/

// We are not doing any internationalization stuff with boost::lexical_cast nor
// are reading numbers like "329,387.38971".
#define BOOST_LEXICAL_CAST_ASSUME_C_LOCALE

#include "gpp_exception.hpp"

#include <limits>
#include <string>

#include <boost/lexical_cast.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"

namespace optimal_learning {

namespace {

/*
  Utility function to append some additional info (file/line number, function name,
  and/or a custom message) to a specified string.
  This is meant to be used for constructing what() messages for the exception classes
  in gpp_exception.hpp.

  INPUTS:
  line_info[]: ptr to char array containing __FILE__ and __LINE__ info; e.g., from OL_STRINGIFY_FILE_AND_LINE
  func_info[]: optional ptr to char array from OL_CURRENT_FUNCTION_NAME or similar
  custom_message[]: optional ptr to char array with any additional text/info to print/log
  message[1]: a valid std::string object
  OUTPUTS:
  message[1]: string with additional info appended
*/
void AppendCustomMessageAndDebugInfo(char const * line_info, char const * func_info, char const * custom_message, std::string * message) {
  if (custom_message) {
    *message += custom_message;
    *message += " ";
  }
  if (func_info) {
    *message += func_info;
    *message += " ";
  }
  *message += line_info;
}

}  // end unnamed namespace

RuntimeException::RuntimeException(char const * line_info, char const * func_info, char const * custom_message) : message_(kName) {
  message_ += ": ";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message, &message_);
}

template <typename ValueType>
BoundsException<ValueType>::BoundsException(char const * name_in, char const * line_info, char const * func_info, char const * custom_message, ValueType value_in, ValueType min_in, ValueType max_in) : value_(value_in), min_(min_in), max_(max_in), message_(name_in) {
  message_ += ": value: " + boost::lexical_cast<std::string>(value_) + " is not in range [" + boost::lexical_cast<std::string>(min_) + ", " + boost::lexical_cast<std::string>(max_) + "]\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message, &message_);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template class BoundsException<int>;
template class BoundsException<double>;

template <typename ValueType>
InvalidValueException<ValueType>::InvalidValueException(char const * line_info, char const * func_info, char const * custom_message, ValueType value_in, ValueType truth_in) : value_(value_in), truth_(truth_in), tolerance_(0), message_(kName) {
  message_ += ": " + boost::lexical_cast<std::string>(value_) + " != " + boost::lexical_cast<std::string>(truth_) + " (value != truth)\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message, &message_);
}

template <typename ValueType>
template <typename ValueTypeIn, typename>
InvalidValueException<ValueType>::InvalidValueException(char const * line_info, char const * func_info, char const * custom_message, ValueType value_in, ValueType truth_in, ValueType tolerance_in) : value_(value_in), truth_(truth_in), tolerance_(tolerance_in), message_(kName) {
  message_ += ": " + boost::lexical_cast<std::string>(value_) + " != " + boost::lexical_cast<std::string>(truth_) + " \u00b1 " + boost::lexical_cast<std::string>(tolerance_) + " (value != truth \u00b1 tolerance)\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message, &message_);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template class InvalidValueException<int>;
template class InvalidValueException<double>;
template InvalidValueException<double>::InvalidValueException(char const * line_info, char const * func_info, char const * custom_message, double value_in, double truth_in, double tolerance_in);

SingularMatrixException::SingularMatrixException(char const * line_info, char const * func_info, char const * custom_message, double const * matrix_in, int num_rows_in, int num_cols_in) : num_rows_(num_rows_in), num_cols_(num_cols_in), matrix_(matrix_in, matrix_in + num_rows_*num_cols_), message_(kName) {
  message_ += ": " + std::to_string(num_rows_) + " x " + std::to_string(num_cols_) + " matrix is singular.\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message, &message_);
}

}  // end namespace optimal_learning
