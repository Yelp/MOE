/*!
  \file gpp_exception.cpp
  \rst
  This file contains definitions for the constructors of the various exception classes in gpp_exception.hpp. These ctors
  generally set the ``message_`` member with some debugging information about what the error is and where it occurred.

  In most cases, we use boost::lexical_cast<std::string> to convert from numbers to strings. std::to_string's formatting
  for floating point types is absolutely terrible (but it works fine for integral types, which is where we use it).
\endrst*/

// We are not doing any internationalization stuff with boost::lexical_cast nor
// are reading numbers like "329,387.38971".
#define BOOST_LEXICAL_CAST_ASSUME_C_LOCALE

#include "gpp_exception.hpp"

#include <limits>
#include <string>

#include <boost/lexical_cast.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"

namespace optimal_learning {

void OptimalLearningException::AppendCustomMessageAndDebugInfo(char const * line_info, char const * func_info,
                                                               char const * custom_message) {
  if (custom_message) {
    message_ += custom_message;
    message_ += " ";
  }
  if (func_info) {
    message_ += func_info;
    message_ += " ";
  }
  message_ += line_info;
}

OptimalLearningException::OptimalLearningException(char const * line_info, char const * func_info,
                                                   char const * custom_message) : message_(kName) {
  message_ += ": ";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message);
}

OptimalLearningException::OptimalLearningException(char const * name) : message_(name) {
}

template <typename ValueType>
BoundsException<ValueType>::BoundsException(char const * name_in, char const * line_info,
                                            char const * func_info, char const * custom_message,
                                            ValueType value_in, ValueType min_in, ValueType max_in)
    : OptimalLearningException(name_in),
      value_(value_in),
      min_(min_in),
      max_(max_in) {
  message_ += ": value: " + boost::lexical_cast<std::string>(value_) + " is not in range [" +
      boost::lexical_cast<std::string>(min_) + ", " + boost::lexical_cast<std::string>(max_) + "]\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message);
}

template <typename ValueType>
BoundsException<ValueType>::BoundsException(char const * line_info, char const * func_info,
                                            char const * custom_message, ValueType value_in,
                                            ValueType min_in, ValueType max_in)
    : BoundsException(kName, line_info, func_info, custom_message, value_in, min_in, max_in) {
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template class BoundsException<int>;
template class BoundsException<double>;

template <typename ValueType>
InvalidValueException<ValueType>::InvalidValueException(char const * line_info, char const * func_info,
                                                        char const * custom_message, ValueType value_in,
                                                        ValueType truth_in)
    : OptimalLearningException(kName), value_(value_in), truth_(truth_in), tolerance_(0) {
  message_ += ": " + boost::lexical_cast<std::string>(value_) + " != " +
      boost::lexical_cast<std::string>(truth_) + " (value != truth)\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message);
}

template <typename ValueType>
template <typename ValueTypeIn, typename>
InvalidValueException<ValueType>::InvalidValueException(char const * line_info, char const * func_info,
                                                        char const * custom_message, ValueType value_in,
                                                        ValueType truth_in, ValueType tolerance_in)
    : OptimalLearningException(kName), value_(value_in), truth_(truth_in), tolerance_(tolerance_in) {
  message_ += ": " + boost::lexical_cast<std::string>(value_) + " != " + boost::lexical_cast<std::string>(truth_) +
      " \u00b1 " + boost::lexical_cast<std::string>(tolerance_) + " (value != truth \u00b1 tolerance)\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template class InvalidValueException<int>;
template class InvalidValueException<double>;
template InvalidValueException<double>::InvalidValueException(
    char const * line_info, char const * func_info, char const * custom_message, double value_in,
    double truth_in, double tolerance_in);

SingularMatrixException::SingularMatrixException(char const * line_info, char const * func_info,
                                                 char const * custom_message, double const * matrix_in,
                                                 int num_rows_in, int leading_minor_index_in)
    : OptimalLearningException(kName), num_rows_(num_rows_in), leading_minor_index_(leading_minor_index_in),
      matrix_(matrix_in, matrix_in + Square(num_rows_)) {
  message_ += ": " + std::to_string(num_rows_) + " x " + std::to_string(num_rows_) + " matrix is singular; " +
      std::to_string(leading_minor_index_) + "-th leading minor is not SPD.\n";
  AppendCustomMessageAndDebugInfo(line_info, func_info, custom_message);
}

}  // end namespace optimal_learning
