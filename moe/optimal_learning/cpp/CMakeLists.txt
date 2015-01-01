# CMakeListsltxt for the optimal_learning project
# Style Notes:
# This tries to follow the spirit of Google's Shell Style Guide.
# There is no notion of "readonly" variables, so these are in all-caps (as per style) and marked with a comment.
#
# Source files are listed at the top of this file. New sources should be added to the appropriate list.
#
# At the top of this file we define several EXTRA_* variables. These variables are meant to be added to each
# target/library via set_target_properties(). These variables represent reasonable defautls; users should feel free
# to alter them (for all targets at once) or modify individual targets.
cmake_minimum_required(VERSION 2.8.9)
# 2.8.8: OBJECT functionality in add_library
#        Workaround: build the OBJECT libraries into STATIC or SHARED libs and proceed accordingly.
# 2.8.9: CMAKE_POSITION_INDEPENDENT_CODE variable (don't have to think about platform-independent way to add -fPIC everywhere)
#        If unavailable, you can add -fPIC (or equivalent) to CMAKE_CXX_FLAGS and remove CMAKE_POSITION_INDEPENDENT_CODE.
#
# TODO(GH-120): Require version 2.8.12. Instead of:
# set_target_properties(
#   OPTIMAL_LEARNING_CORE_BUNDLE PROPERTIES
#   COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
#   COMPILE_DEFINITIONS "${EXTRA_COMPILE_DEFINITIONS}"
#   )
# the new cmake lets us do:
#   target_compile_definitions(OPTIMAL_LEARNING_CORE_BUNDLE PUBLIC "${EXTRA_COMPILE_DEFINITIONS}")
# and so on for options/flags, includes, etc. We already do this with target_link_libraries(), and 2.8.12 just
# lets us make the usage more uniform. This gives more flexibility and gets rid of the clunky
# set_target_properties() interface (unclear overwrite behavior, list expansion, etc).
# It also lets us specify separate public/private interfaces and delineate which options/links were used in compilation.
project(optimal_learning)

#### Build type
# Why wouldn't you always want the -g flag?
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

#### User configured external dependencies
# Sometimes cmake has difficulty finding Python (e.g., multiple installations, especially on OS X with
# system vs macport/homebrew versions); explicitly specify the location of Python when that happens.
# See documentation in FindPythonLibs.cmake for more information.

# User specifies what directory to look for Python headers in
if (DEFINED MOE_PYTHON_INCLUDE_DIR)
  if (IS_DIRECTORY ${MOE_PYTHON_INCLUDE_DIR})
    set(PYTHON_INCLUDE_DIR
      ${MOE_PYTHON_INCLUDE_DIR}
      CACHE PATH "*Full* path to where Python.h is found."
      )
  else()
    message(FATAL_ERROR "MOE_PYTHON_INCLUDE_DIR is not a valid directory: " ${MOE_PYTHON_INCLUDE_DIR})
  endif()
endif()

# User specifies the location of the Python dynamic library
if (DEFINED MOE_PYTHON_LIBRARY)
  if (EXISTS ${MOE_PYTHON_LIBRARY} AND NOT IS_DIRECTORY ${MOE_PYTHON_LIBRARY})
    set(PYTHON_LIBRARY
      ${MOE_PYTHON_LIBRARY}
      CACHE FILEPATH "*Full* path to Python shared object (e.g., libpython2.7.so or .dylib)."
      )
  else()
    message(FATAL_ERROR "MOE_PYTHON_LIBRARY is not a valid file: " ${MOE_PYTHON_LIBRARY})
  endif()
endif()

# User specifies priority Python versions to look for (in decreasing order of priority)
if (DEFINED MOE_PYTHON_ADDITIONAL_VERSIONS)
  set(Python_ADDITIONAL_VERSIONS
    ${MOE_PYTHON_ADDITIONAL_VERSIONS}
    CACHE STRING "List (semicolon-separated, NO SPACES) of Python version(s), ranked by priority; e.g., 2.7;2.6;3.2"
    )
endif()

#### External dependencies
find_package(PythonLibs REQUIRED)
find_package(Boost 1.51 COMPONENTS python REQUIRED)

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#### Sources
# Lists of source files that are dependencies. Currently we have one group for "core" functionality,
# one group for their test files, and one group for the C++<->Python interface.
# Do not include files with main() or BOOST_PYTHON_MODULE since no sources depend on those files.
# readonly
set(OPTIMAL_LEARNING_CORE_SOURCES
  gpp_covariance.cpp
  gpp_domain.cpp
  gpp_exception.cpp
  gpp_heuristic_expected_improvement_optimization.cpp
  gpp_linear_algebra.cpp
  gpp_logging.cpp
  gpp_math.cpp
  gpp_model_selection.cpp
  gpp_random.cpp
  gpp_expected_improvement_gpu.cpp
  )

# readonly
set(OPTIMAL_LEARNING_TEST_SOURCES
  gpp_covariance_test.cpp
  gpp_domain_test.cpp
  gpp_geometry_test.cpp
  gpp_heuristic_expected_improvement_optimization_test.cpp
  gpp_linear_algebra_test.cpp
  gpp_math_test.cpp
  gpp_model_selection_test.cpp
  gpp_optimization_test.cpp
  gpp_random_test.cpp
  gpp_test_utils.cpp
  gpp_test_utils_test.cpp
  gpp_expected_improvement_gpu_test.cpp
  )

# readonly
set(OPTIMAL_LEARNING_PYTHON_SOURCES
  gpp_python.cpp
  gpp_python_common.cpp
  gpp_python_expected_improvement.cpp
  gpp_python_gaussian_process.cpp
  gpp_python_model_selection.cpp
  gpp_python_test.cpp
  )

# readonly
set(DEMO_NAMES
  demo_hyper
  demo_EI
  demo_full
  )

# readonly
set(DEMO_SOURCES
  gpp_hyperparameter_optimization_demo.cpp
  gpp_expected_improvement_demo.cpp
  gpp_hyper_and_EI_demo.cpp
  )

#### Extra flags and definitions
# These are meant to be reasonable defaults. Users should feel free to alter the definitions (changing
# all targets) or append to individual targets as desired.

#### Extra link flags
# e.g., add -fprofile-arcs -ftest-coverage to use gcov
# This list need to be quoted.
# readonly
set(EXTRA_LINK_FLAGS "")

#### Extra compiler flags
# This list need to be quoted.
# readonly
set(EXTRA_COMPILE_FLAGS " -march=native")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # Remove -march=native to prevent gcc from generating AVX (and newer) x86 extension instructions. OS X's GNU
  # assembler (as) is too old to parse these; they have been on Intel hardware since 2008 and AMD since 2011.
  # Power users: this can be overcome by replacing the program "as" with a script to invoke clang's assembler.
  # This can be valuable: performance gains from this one flag may be some 10-15%. Script example:
  # http://mac-os-forge.2317878.n4.nabble.com/gcc-as-AVX-binutils-and-MacOS-X-10-7-tp144472p144475.html
  string(REGEX REPLACE "-march=native" "" EXTRA_COMPILE_FLAGS ${EXTRA_COMPILE_FLAGS})
endif()

# Set some reasonable default extra flags for various compilers.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -ftree-vectorize")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -restrict -fstrict-aliasing -alias-const -ftz -no-prec-div -no-prec-sqrt -fast-transcendentals -opt-subscript-in-range -fp-model fast=1")
endif()

#### Extra compiler definitions
# Do not quote these lists.
# readonly
set(EXTRA_COMPILE_DEFINITIONS_WARNING OL_WARNING_PRINT=1)
# readonly
set(EXTRA_COMPILE_DEFINITIONS_DEBUG OL_DEBUG_PRINT=1)

# readonly
set(EXTRA_COMPILE_DEFINITIONS "")
if(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  set(EXTRA_COMPILE_DEFINITIONS ${EXTRA_COMPILE_DEFINITIONS_WARNING})
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(EXTRA_COMPILE_DEFINITIONS ${EXTRA_COMPILE_DEFINITIONS_DEBUG})
endif()

# Sets up an object library target. Object libraries essentially create a subdirectory with a
# "%.o" target in a Makefile. That is, it sets up targets for each specified source file but
# does not produce an archive. The resulting "library" is not linked, but it can be used as
# a dependency for other "real" cmake targets. This saves compile time.
# See: http://www.cmake.org/Wiki/CMake/Tutorials/Object_Library
#
# INPUTS
# name: the name of the object library; cmake will produce a subdirectory with this name
#       holding the object files and you refer to this name when specifying dependencies
# sources: list of source files to be compiled into the object library
# compile_flags: list of additional compilation flags to use
# compile_definitions: list of additional compilation definitions to use (DO NOT QUOTE THIS!)
function(configure_object_library name sources compile_flags compile_definitions)
  add_library(
    ${name} OBJECT
    ${sources}
    )
  set_target_properties(
    ${name} PROPERTIES
    COMPILE_FLAGS "${compile_flags}"
    COMPILE_DEFINITIONS "${compile_definitions}"
    )
endfunction(configure_object_library)

# Adds cmake executable targets for all specified name, source pairs.
# Each "source" file must have a main() function.
#
# Note: this creates object libraries for each executable source; for a source
#       named "foo.cpp", the object lib will be called "foo".
#
# INPUTS:
# exec_names: a list of names of executables
# exec_sources: a list of sources with main()
# compile_flags: list of additional compilation flags to use
# compile_definitions: list of additional compilation definitions to use (DO NOT QUOTE THIS!)
function(configure_exec_targets exec_names exec_sources dependencies compile_flags compile_definitions)
  list(LENGTH exec_names num_demos)
  math(EXPR num_demos "${num_demos} - 1")
  foreach(index RANGE ${num_demos})
    list(GET exec_names ${index} name)
    list(GET exec_sources ${index} source)

    # Drop the ".cpp" file ending to produce the library name.
    string(REGEX REPLACE "\\.cpp" "" object_name ${source})

    # C/C++'s ODR means we can only have one executable src per object lib.
    # This is a performance optimization: putting the excutable srcs in object
    # libs allows them to be compiled in parallel with all other source files.
    configure_object_library(
      ${object_name}
      ${source}
      "${compile_flags}"
      "${compile_definitions}"
      )

    add_executable(
      ${name}
      $<TARGET_OBJECTS:${object_name}>
      ${dependencies}
      )

    set_target_properties(
      ${name} PROPERTIES
      PREFIX ""
      COMPILE_FLAGS "${compile_flags}"
      COMPILE_DEFINITIONS "${compile_definitions}"
      LINK_FLAGS "${EXTRA_LINK_FLAGS}"
      )
    if (${MOE_USE_GPU} MATCHES "1")
        add_dependencies(${name} GPU_LIB)
        target_link_libraries(${name} ${CUDA_LIBRARIES}
            ${CMAKE_BINARY_DIR}/gpu/libOL_GPU.so)
    endif()
  endforeach()
endfunction(configure_exec_targets)

#### Basic compiler flags
# Enable position independent code (gcc: -fPIC) by default.
# PIC is safer (code doesn't have to be located in the same place in memory every time) and all but
# required when generating 64-bit shared libraries.
# In x86-64, the well-known penalty from the x86 (32bit) days is all but gone; the addition of the new
# Relative Instruction Pointer addressing mode eliminates much of the previous overhead.
# http://eli.thegreenplace.net/2011/11/11/position-independent-code-pic-in-shared-libraries-on-x64/
# http://www.technovelty.org/c/position-independent-code-and-x86-64-libraries.html
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define some extra warnings (not enabled by -Wall -Wextra) for C/C++ that are nonetheless useful for guarding
# against potential errors, bad style, etc.
# Note: -Weffc++ is a useful one to check periodically, but its false positive rate is annoying.
# TODO(GH-151): With later versions of gcc, add: -Wuseless-cast -Wdate-time -Wfloat-conversion
set(EXTRA_C_WARNINGS " -Wundef -Wshadow -Wswitch-default -Wformat-nonliteral -Wpointer-arith -Wcast-align -Wcast-qual -Wwrite-strings -Wlogical-op")
set(EXTRA_C++_WARNINGS " -Wold-style-cast -Wnon-virtual-dtor -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wzero-as-null-pointer-constant -Wnoexcept -Wstrict-null-sentinel")

# Set baseline C++ flags; these should be good for everyone. The first two (C++11, OpenMP) are required.
# Set C++11
# Enable OpenMP
# Add extra warnings to guard against common error sources/bad style.
# Add (gcc) math flags; these strictly improve performance with no loss of "safety."
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -Wall -Wextra ${EXTRA_C_WARNINGS} ${EXTRA_C++_WARNINGS} -fno-trapping-math -fno-signaling-nans -fno-math-errno")

# O3 takes longer to compile and the code produced is at best no faster than O2 (gcc, icc).
string(REGEX REPLACE "O3" "O2" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

#### GPU Component
# readonly
set(EXTRA_COMPILE_DEFINITIONS_GPU OL_GPU_ENABLED)

# CUDA C compiler may be different from the c/c++ compiler used for compiling cpp
# code, because current available CUDA version does not support gcc 4.7+
set(GPU_CC ${CMAKE_C_COMPILER})
set(GPU_CXX ${CMAKE_CXX_COMPILER})

if (DEFINED MOE_GPU_CC)
  if (EXISTS ${MOE_GPU_CC} AND NOT IS_DIRECTORY ${MOE_GPU_CC})
    set(GPU_CC ${MOE_GPU_CC})
  else()
    message(FATAL_ERROR "MOE_GPU_CC is not a valid file: " ${MOE_GPU_CC})
  endif()
endif()

if (DEFINED MOE_GPU_CXX)
  if (EXISTS ${MOE_GPU_CXX} AND NOT IS_DIRECTORY ${MOE_GPU_CXX})
    set(GPU_CXX ${MOE_GPU_CXX})
  else()
    message(FATAL_ERROR "MOE_GPU_CXX is not a valid file: " ${MOE_GPU_CXX})
  endif()
endif()

# If MOE_USE_GPU is turned on via MOE_CMAKE_OPTS, cmake will try to find CUDA
# package and call gpu cmake to build CUDA code
if (${MOE_USE_GPU} MATCHES "1")
    if (DEFINED MOE_CUDA_SDK_INCLUDE_DIRS)
        if (NOT IS_DIRECTORY ${MOE_CUDA_SDK_INCLUDE_DIRS})
            message(FATAL_ERROR "MOE_CUDA_SDK_INCLUDE_DIRS is not a valid directory: " ${MOE_CUDA_SDK_INCLUDE_DIRS})
        endif()
    else()
        message(FATAL_ERROR "MOE_CUDA_SDK_INCLUDE_DIRS not set!")
    endif()

    find_package(CUDA 5.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})

    set(EXTRA_COMPILE_DEFINITIONS ${EXTRA_COMPILE_DEFINITIONS}
       ${EXTRA_COMPILE_DEFINITIONS_GPU})
    set(MOE_GPU_CMAKE_OPTS
        "-D MOE_CUDA_SDK_INCLUDE_DIRS=${MOE_CUDA_SDK_INCLUDE_DIRS}")

    add_custom_target(
        GPU_FOLDER
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/gpu/
        )

    add_custom_target(
        GPU_LIB
        DEPENDS GPU_FOLDER
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gpu/
        COMMAND CC=${GPU_CC} CXX=${GPU_CXX} cmake `echo
        ${MOE_GPU_CMAKE_OPTS}` ${CMAKE_SOURCE_DIR}/gpu/
        COMMAND make
        )
endif()

#### Object libraries
# See configure_object_library() function comments for more details.
# WARNING: You MUST have compatible flags set between OBJECT libraries and targets that depend on them!
# e.g., not setting -fPIC in OBJECT libraries makes then unusable in shared libs, setting -fpack-struct
# in one place but not another leads to binary incompatibility, etc.
configure_object_library(
  OPTIMAL_LEARNING_CORE_BUNDLE
  "${OPTIMAL_LEARNING_CORE_SOURCES}"
  "${EXTRA_COMPILE_FLAGS}"
  "${EXTRA_COMPILE_DEFINITIONS}"
  )
configure_object_library(
  OPTIMAL_LEARNING_TEST_BUNDLE
  "${OPTIMAL_LEARNING_TEST_SOURCES}"
  "${EXTRA_COMPILE_FLAGS}"
  "${EXTRA_COMPILE_DEFINITIONS}"
  )
configure_object_library(
  OPTIMAL_LEARNING_PYTHON_BUNDLE
  "${OPTIMAL_LEARNING_PYTHON_SOURCES}"
  "${EXTRA_COMPILE_FLAGS}"
  "${EXTRA_COMPILE_DEFINITIONS}"
  )

#### "GPP" library: for connecting Python to C++
# Make an empty __init__.py file to turn the build directory into a package.
# This file is required to do "import path.to.library.GPP" in Python.
add_custom_command(
  OUTPUT __init__.py
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo 'r"""Built C++ files."""' > __init__.py
  )

# Rule for GPP library; includes dependency on having __init__.py in the same directory
add_library(
  GPP SHARED
  $<TARGET_OBJECTS:OPTIMAL_LEARNING_PYTHON_BUNDLE>
  $<TARGET_OBJECTS:OPTIMAL_LEARNING_CORE_BUNDLE>
  $<TARGET_OBJECTS:OPTIMAL_LEARNING_TEST_BUNDLE>
  ${CMAKE_BINARY_DIR}/__init__.py
  )
if (${MOE_USE_GPU} MATCHES "1")
    add_dependencies(GPP GPU_LIB)
endif()

target_link_libraries(GPP ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
if (${MOE_USE_GPU} MATCHES "1")
    target_link_libraries(GPP ${CUDA_LIBRARIES} ${CMAKE_BINARY_DIR}/gpu/libOL_GPU.so)
endif()
        
# Make sure lib ends in ".so"
set_target_properties(
  GPP PROPERTIES
  PREFIX ""
  SUFFIX .so
  COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
  COMPILE_DEFINITIONS "${EXTRA_COMPILE_DEFINITIONS}"
  LINK_FLAGS "${EXTRA_LINK_FLAGS}"
  )

#### Demo executables
set(dependencies $<TARGET_OBJECTS:OPTIMAL_LEARNING_CORE_BUNDLE> gpp_test_utils.cpp)
configure_exec_targets(
  "${DEMO_NAMES}"
  "${DEMO_SOURCES}"
  "${dependencies}"
  "${EXTRA_COMPILE_FLAGS}"
  "${EXTRA_COMPILE_DEFINITIONS}"
  )

# Dummy target named "demos" that builds all demos
add_custom_target(demos)
add_dependencies(demos ${DEMO_NAMES})
