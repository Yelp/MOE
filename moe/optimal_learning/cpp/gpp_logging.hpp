// gpp_logging.hpp
/*
  This file contains some macros wrapping std::printf(). Currently, the "log file" is just stdout. And
  the verbosity level must be chosen at compile-time, although we do provide separate macros for
  debug, verbose, warning, and error.

  We also have utilities for printing commonly used structures in the code. Currently this file contains
  a printer for an array of ClosedInterval. Printers for vectors/matrices will be moved in #60254.

  Long term, this file will connect to a real logging library (see TODO below).

  TODO(eliu): (#48960) connect MOE/C++ to a real logging library instead of using stdout as our log.

  TODO(eliu): (#59445) rename macros to be prefixed with "OL_" to 'namespace' them

  TODO(eliu): (#60254) Move PrintMatrix() and PrintMatrixTrans() functions here from gpp_linear_algebra
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_LOGGING_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_LOGGING_HPP_

#include <cstdio>

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Macro wrapper for printf so it can be easily disabled via the compiler option OL_DEBUG_PRINT.
  This is meant to give extra details about internal workings/state of code.
*/
#ifdef OL_DEBUG_PRINT
#define OL_DEBUG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define OL_DEBUG_PRINTF(...) (void)0
#endif

/*
  Macro wrapper for printf so it can be easily disabled via the compiler optionOL_VERBOSE_PRINT.
  This is meant to provide extra information about convergence characteristics, etc.
*/
#ifdef OL_VERBOSE_PRINT
#define OL_VERBOSE_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define OL_VERBOSE_PRINTF(...) (void)0
#endif

#if defined(OL_VERBOSE_PRINT) || defined(OL_DEBUG_PRINT)
#ifndef OL_WARNING_PRINT
#define OL_WARNING_PRINT
#endif
#endif

/*
  Macro wrapper for printf so it can be easily disabled via the compiler option OL_WARNING_PRINT.
  This is meant for printing messages that are warnings--something went wrong, but the error
  is not severe enough to warrant exiting.
*/
#ifdef OL_WARNING_PRINT
#define OL_WARNING_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define OL_WARNING_PRINTF(...) (void)0
#endif

#ifdef OL_WARNING_PRINT
#ifndef OL_ERROR_PRINT
#define OL_ERROR_PRINT
#endif
#endif

/*
  Macro wrapper for printf so it can be easily disabled via the compiler option OL_ERROR_PRINT.
  This is meant for printing messages that are errors--something went catastrophically wrong.
*/
#ifdef OL_ERROR_PRINT
#define OL_ERROR_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define OL_ERROR_PRINTF(...) (void)0
#endif

/*
  Macros for printf'ing in color.

  Inserting these macros before a string component of printf() changes the color
  of printf until a new color or reset is specified.  These macros can be
  interspersed between strings in printf()'s first argument (see examples).

  Example usage:
  printf(OL_ANSI_COLOR_GREEN "Hi I am %d" OL_ANSI_COLOR_RESET "And you are %d\n", 2, 3);
  Would print:
  Hi I am 2 And you are 3.
  |  GREEN | DEFAAULT COLOR|

  Without reset, the following would happen:
  printf(OL_ANSI_COLOR_GREEN "Hi I am %d\n", 2);
  printf("And you are %d\n", 3);
  Hi I am 2     <--- in green
  And you are 3 <--- ALSO in green!
*/
#define OL_ANSI_COLOR_RESET   "\x1b[0m"                  // reset to default color
#define OL_ANSI_COLOR_RED     "\x1b[31m"                 // Red
#define OL_ANSI_COLOR_BLACK   "\033[30m"                 // Black
#define OL_ANSI_COLOR_GREEN   "\x1b[32m"                 // Green
#define OL_ANSI_COLOR_YELLOW  "\x1b[33m"                 // Yellow
#define OL_ANSI_COLOR_BLUE    "\x1b[34m"                 // Blue
#define OL_ANSI_COLOR_MAGENTA "\x1b[35m"                 // Magenta
#define OL_ANSI_COLOR_CYAN    "\x1b[36m"                 // Cyan
#define OL_ANSI_COLOR_WHITE   "\033[37m"                 // White
#define OL_ANSI_COLOR_BOLDBLACK   "\033[1m\033[30m"      // Bold Black
#define OL_ANSI_COLOR_BOLDRED     "\033[1m\033[31m"      // Bold Red
#define OL_ANSI_COLOR_BOLDGREEN   "\033[1m\033[32m"      // Bold Green
#define OL_ANSI_COLOR_BOLDYELLOW  "\033[1m\033[33m"      // Bold Yellow
#define OL_ANSI_COLOR_BOLDBLUE    "\033[1m\033[34m"      // Bold Blue
#define OL_ANSI_COLOR_BOLDMAGENTA "\033[1m\033[35m"      // Bold Magenta
#define OL_ANSI_COLOR_BOLDCYAN    "\033[1m\033[36m"      // Bold Cyan
#define OL_ANSI_COLOR_BOLDWHITE   "\033[1m\033[37m"      // Bold White

// for top-level tests, matching testify's color output
#define OL_SUCCESS_PRINTF(...) std::printf(OL_ANSI_COLOR_BOLDGREEN "SUCCESS: " OL_ANSI_COLOR_RESET __VA_ARGS__)
#define OL_FAILURE_PRINTF(...) std::printf(OL_ANSI_COLOR_BOLDRED "FAILURE: " OL_ANSI_COLOR_RESET __VA_ARGS__)
// for test sub-components
#define OL_PARTIAL_SUCCESS_PRINTF(...) std::printf(OL_ANSI_COLOR_GREEN "ok: " OL_ANSI_COLOR_RESET __VA_ARGS__)
#define OL_PARTIAL_FAILURE_PRINTF(...) std::printf(OL_ANSI_COLOR_RED "fail: " OL_ANSI_COLOR_RESET __VA_ARGS__)

struct ClosedInterval;

/*
  Prints an array of ClosedInterval, formatted as [x_i_min, x_i_max] for i = 0 .. dim-1, one interval per line.

  INPUTS:
  domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
  dim_in: number of spatial dimensions
*/
void PrintDomainBounds(ClosedInterval const * restrict domain_bounds, int dim);

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_LOGGING_HPP_
