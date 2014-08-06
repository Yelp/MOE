/*!
  \file gpp_python.cpp
  \rst
  This file contains the "call" to ``BOOST_PYTHON_MODULE``; think of that as the ``main()`` function for the interface.
  It includes the full docstring for the Python module. That call wraps ``Export.*()`` functions from ``gpp_python_.*``
  helper files, which contain the pieces of ``C++`` functionality that we are exporting to Python (e.g., debugging
  tools like GP mean, variance, and gradients as well as EI optimizers and model selection tools).

  This file also includes the logic for translating C++ exceptions to Python exceptions.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <exception>  // NOLINT(build/include_order)
#include <string>  // NOLINT(build/include_order)
#include <type_traits>  // NOLINT(build/include_order)

#include <boost/python/handle.hpp>  // NOLINT(build/include_order)
#include <boost/python/docstring_options.hpp>  // NOLINT(build/include_order)
#include <boost/python/errors.hpp>  // NOLINT(build/include_order)
#include <boost/python/exception_translator.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/module.hpp>  // NOLINT(build/include_order)
#include <boost/python/object.hpp>  // NOLINT(build/include_order)
#include <boost/python/scope.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_python_common.hpp"
#include "gpp_python_expected_improvement.hpp"
#include "gpp_python_gaussian_process.hpp"
#include "gpp_python_model_selection.hpp"
#include "gpp_python_test.hpp"

namespace optimal_learning {

namespace {  // unnamed namespace for exception translation (for BOOST_PYTHON_MODULE(GPP))

/*!\rst
  Helper function to build a Python exception type object called "name" within the specified scope.
  The new Python exception will subclass ``Exception``.

  Afterward, ``scope`` will have a new callable object called ``name`` that can be used to
  construct an exception instance. For example, if ``name = "MyPyException"``, then in Python::

    >>> import scope
    >>> raise scope.MyPyException("my message")

  .. WARNING:: ONLY call this function from within a ``BOOST_PYTHON_MODULE`` block since it only has
    meaning during module construction. After module construction, scope has no meaning and it
    will probably be a dangling pointer or NoneType, leading to an exception or segfault.

  \param
    :name[]: ptr to char array containing the desired name of the new Python exception type
    :docstring[]: ptr to char array with the docstring for the new Python exception type
    :base_type[1]: nullptr or ptr to the Python type object serving as the base class for the new exception.
      nullptr means PyExc_Exception (Python type Exception) will be used.
    :scope[1]: the scope to add the new exception types to
  \output
    :base_type[1]: python object modified through PyErr_NewException call (incref'd).
    :scope[1]: the input scope with the new exception types added
  \return
    PyObject pointer to the (callable) type object (the new exception type) that was created
\endrst*/
OL_WARN_UNUSED_RESULT PyObject * CreatePyExceptionClass(const char * name, const char * docstring, PyObject * base_type, boost::python::scope * scope) {
  std::string scope_name = boost::python::extract<std::string>(scope->attr("__name__"));
  std::string qualified_name = scope_name + "." + name;

  /*
    QUESTION FOR REVIEWER: the docs for PyErr_NewException:
    http://docs.python.org/3/c-api/exceptions.html
    claim that it returns a "new reference."
    The meaning of that phrase:
    http://docs.python.org/release/3.3.3/c-api/intro.html#objects-types-and-reference-counts
    First para: "When a function passes ownership of a reference on to its caller, the caller is said to receive a new reference"

    BUT in this example:
    http://docs.python.org/3.3/extending/extending.html#intermezzo-errors-and-exceptions
    (scroll down to the first code block, reading "Note also that the SpamError variable retains...")
    They set::

      SpamError = PyErr_NewException(...);  (1)
      Py_INCREF(SpamError);                 (2)

    So ``PyErr_NewException`` returns a *new reference* that SpamError owns in (1). Then in (2), SpamError owns
    the reference... again?!  (This example appears unchanged since somewhere in python 1.x; maybe it's just old.)
    It seems like this ``Py_INCREF`` call is just a 'safety buffer'?

    The doc language (new reference) would lead me to believe that SpamError owns a reference to the new type object.
    When SpamError is done, it should be ``Py_DECREF``'d. (Unless SpamError is never done--which is appropriate here, I
    believe, since type objects should not be deallocated as I have no control of whether all referrers have
    been destroyed.)

    So... is my way/interpretation right? Or should I follow python's example?

    This is not a critical detail. Having too many INCREFs just makes the thing "more" immortal.
    Still I'd like to square this away correctly.
  */
  // const_cast: PyErr_NewExceptionWithDoc expects char *, not char const *. This is an ommission:
  // http://bugs.python.org/issue4949
  // first nullptr: base class for NewException will be PyExc_Exception unless overriden
  // second nullptr: no default member fields (could pass a dict with default fields)
#if defined(PY_MAJOR_VERSION) && ((PY_MAJOR_VERSION > 2) || ((PY_MAJOR_VERSION == 2) && (PY_MINOR_VERSION >= 7)))
  PyObject * type_object = PyErr_NewExceptionWithDoc(const_cast<char *>(qualified_name.c_str()), const_cast<char *>(docstring), base_type, nullptr);
#else
  // PyErr_NewExceptionWithDoc did not exist before Python 2.7, so we "cannot" attach a docstring to our new type object.
  // Attributes of metaclassses (i.e., type objects) are not writeable after creation of that type object. So the only
  // way to add a docstring here would be to build the type from scratch, which is too much pain for just a docstring.
  (void) docstring;  // quiet the compiler warning (unused variable)
  PyObject * type_object = PyErr_NewException(const_cast<char *>(qualified_name.c_str()), base_type, nullptr);
#endif
  if (!type_object) {
    boost::python::throw_error_already_set();
  }
  scope->attr(name) = boost::python::object(boost::python::handle<>(boost::python::borrowed(type_object)));

  return type_object;
}

/*!\rst
  When translating ``C++`` exceptions to Python exceptions, we need to identify a base class. By Python convention,
  we want these to be Python types inheriting from Exception.
  This is a monostate object that knows how to set up these base classes in Python; it also keeps pointers to
  the resulting Python type objects for future use (e.g., instantiating Python exceptions).

  .. NOTE:: this class follows the Monostate pattern, implying GLOBAL STATE.

  http://c2.com/cgi/wiki?MonostatePattern
  http://www.informit.com/guides/content.aspx?g=cplusplus&seqNum=147
  That is, the various PyObject * for Python type objects (and others) are stored as private, static variables.

  We use monostate because once the Python type objects have been created, we need to hold on to references to
  them until the end of time. Type objects must never be deallocated:
  http://docs.python.org/release/3.3.3/c-api/intro.html#objects-types-and-reference-counts

  "The sole exception are the type objects; since these must never be deallocated, they are typically static PyTypeObject objects."

  Also, see python/object.h's header comments (object.h, Python 2.7):

  "Type objects are exceptions to the first rule; the standard types are represented by

  statically initialized type objects, although work on type/class unification
  for Python 2.2 made it possible to have heap-allocated type objects too."

  Thus, as long as the enclosing Python module lives, we need to hold references to the (exception) type objects,
  which (as far as I know) requires global state. So storing the type objects in boost::python::object (which could
  be destructed) is not an option. We could also heap allocate this container and never delete,
  but that seems even more confusing. Besides, static variables is how it is done in Python.

  Additionally, this class protects against redundant calls to ``PyErr_NewException`` (which creates exception type objects).
  Redundant here means creating multiple exception types with the same name.  Failing to do so would add MULTIPLE
  instances of the "same" type (Python has no ODR), which is confusing to the user.

  .. NOTE:: we cannot mark PyObject * pointers as pointers to const, e.g., ::

      const PyObject * some_object;

  because Python C-API calls will modify the objects. HOWEVER, DO NOT CHANGE these pointers and
  DO NOT CHANGE the things they point to!  (Outside of Python calls that is.)
  The pointers in the monostate are meant to be "conceptually" const.
\endrst*/
class PyExceptionClassContainer {
 public:
  /*!\rst
    Initializes the (mono-) state. This defines new python type objects (for exceptions) and saves the
    python scope that they are defined in.

    If Initialize() was already called previously (with no subsequent calls to Reset()), this
    function does nothing, preventing defining multiple "identical" types in Python.

    If Initialize() is not called at program start or after Reset(), all newly translated exceptions will
    be of type: default_exception_type_object_.

    .. WARNING:: NOT THREAD SAFE in C++. It might be thread-safe in Python calls; not sure
      how the GIL is handled here.
      However there is no reason to call this from multiple threads in C++ so I'm ignoring the issue.

    \param
      :scope[1]: the scope to add the new exception types to
    \output
      :scope[1]: the input scope with the new exception types added
  \endrst*/
  void Initialize(boost::python::scope * scope) OL_NONNULL_POINTERS {
    // Prevent duplicate definitions (in Python) of the same objects
    if (!initialized_) {
      scope_ = scope;

      // Note: If listing exception type objects here gets unwieldly, we can store them in an
      // array<> of typle<>, making Initialize() just a simple for (item : array) { ... } loop.
      static char const * optimal_learning_exception_docstring = "Base exception class for errors thrown/raised from the (C++) ``optimal_learning`` library.";
      optimal_learning_exception_type_object_ = CreatePyExceptionClass(OptimalLearningException::kName, optimal_learning_exception_docstring, nullptr, scope_);

      static char const * bounds_exception_docstring = "value not in range [min, max].";
      bounds_exception_type_object_ = CreatePyExceptionClass(BoundsException<double>::kName, bounds_exception_docstring, optimal_learning_exception_type_object_, scope_);

      static char const * invalid_value_exception_docstring = "value != truth (+/- tolerance)";
      invalid_value_exception_type_object_ = CreatePyExceptionClass(InvalidValueException<double>::kName, invalid_value_exception_docstring, optimal_learning_exception_type_object_, scope_);

      static char const * singular_matrix_exception_docstring = "num_rows X num_cols matrix is singular";
      singular_matrix_exception_type_object_ = CreatePyExceptionClass(SingularMatrixException::kName, singular_matrix_exception_docstring, optimal_learning_exception_type_object_, scope_);

      initialized_ = true;
    }
  }

  /*!\rst
    Reset the state back to default. Afer this call, translated exceptions will be of type
    default_exception_type_object_. Future calls to Initialize() will define new exception
    types in another (not necessarily different) scope.

    This is not recommended.

    .. WARNING:: NOT THREAD SAFE. See Initialize() comments.

    .. WARNING:: This makes the existing ``PyObject``s *unreachable* from ``C++``.
      It is unsafe to ``DECREF`` our ``PyObject`` pointers; we cannot guarantee that these type
      objects will outlive all instances. (*I think*)
  \endrst*/
  void Reset() {
    initialized_ = false;
    scope_ = nullptr;
    bounds_exception_type_object_ = default_exception_type_object_;
    invalid_value_exception_type_object_ = default_exception_type_object_;
    singular_matrix_exception_type_object_ = default_exception_type_object_;
  }

  PyObject * optimal_learning_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return optimal_learning_exception_type_object_;
  }

  PyObject * bounds_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return bounds_exception_type_object_;
  }

  PyObject * invalid_value_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return invalid_value_exception_type_object_;
  }

  PyObject * singular_matrix_exception_type_object() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return singular_matrix_exception_type_object_;
  }

 private:
  // Fall back to this type object if something has not been initialized or we are otherwise confused.
  static PyObject * const default_exception_type_object_;

  // pointers to Python callable objects that build the Python exception classes
  static PyObject * optimal_learning_exception_type_object_;
  static PyObject * bounds_exception_type_object_;
  static PyObject * invalid_value_exception_type_object_;
  static PyObject * singular_matrix_exception_type_object_;

  // scope that these exception objects will live in
  static boost::python::scope * scope_;

  // whether this class has been properly initialized
  static bool initialized_;
};

PyObject * const PyExceptionClassContainer::default_exception_type_object_ = PyExc_RuntimeError;
PyObject * PyExceptionClassContainer::optimal_learning_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
PyObject * PyExceptionClassContainer::bounds_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
PyObject * PyExceptionClassContainer::invalid_value_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
PyObject * PyExceptionClassContainer::singular_matrix_exception_type_object_ = PyExceptionClassContainer::default_exception_type_object_;
boost::python::scope * PyExceptionClassContainer::scope_ = nullptr;
bool PyExceptionClassContainer::initialized_ = false;

/*!\rst
  Translate std::exception to a Python exception, maintaining the data fields.
  This translator can capture *any* std::exception, so make sure it does not mask your other translators.

  \param
    :except: C++ exception to translate
    :py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
      translated Python exception base class. We assume that this type inherits from Exception.
  \return
    **NEVER RETURNS**
\endrst*/
OL_NORETURN void TranslateStdException(const std::exception& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.optimal_learning_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  // Note: SetObject gets ownership (*not* borrow) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*!\rst
  Translate BoundsException to a Python exception, maintaining the data fields.

  \param
    :except: C++ exception to translate
    :py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
      translated Python exception base class. We assume that this type inherits from Exception.
  \return
    **NEVER RETURNS**
\endrst*/
template <typename ValueType>
OL_NORETURN void TranslateBoundsException(const BoundsException<ValueType>& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.bounds_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("value") = except.value();
  instance.attr("min") = except.min();
  instance.attr("max") = except.max();

  // Note: SetObject gets ownership (*not* borrow) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*!\rst
  Translate InvalidValueException to a Python exception, maintaining the data fields.

  \param
    :except: C++ exception to translate
    :py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
      translated Python exception base class. We assume that this type inherits from Exception.
  \return
    **NEVER RETURNS**
\endrst*/
template <typename ValueType>
OL_NORETURN void TranslateInvalidValueException(const InvalidValueException<ValueType>& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.invalid_value_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("value") = except.value();
  instance.attr("truth") = except.truth();
  instance.attr("tolerance") = except.tolerance();

  // Note: SetObject gets ownership (*not* borrow) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*!\rst
  Translate SingularMatrixException to a Python exception, maintaining the data fields.

  \param
    :except: C++ exception to translate
    :py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
      translated Python exception base class. We assume that this type inherits from Exception.
  \return
    **NEVER RETURNS**
\endrst*/
OL_NORETURN void TranslateSingularMatrixException(const SingularMatrixException& except, const PyExceptionClassContainer& py_exception_type_objects) {
  PyObject * base_exception_class = py_exception_type_objects.singular_matrix_exception_type_object();

  boost::python::object base_except_type(boost::python::handle<>(boost::python::borrowed(base_exception_class)));
  boost::python::object instance = base_except_type(except.what());  // analogue of PyObject_CallObject(base_except.ptr(), args)

  instance.attr("num_rows") = except.num_rows();
  // TODO(GH-159): this would make more sense as a numpy array/matrix
  instance.attr("matrix") = VectorToPylist(except.matrix());

  // Note: SetObject gets ownership (*not* borrow/steal) of both references (type object and instance/value);
  // i.e., it INCREFs at the start and objects will be DECREF'd when the exception handling completes.
  PyErr_SetObject(base_except_type.ptr(), instance.ptr());
  boost::python::throw_error_already_set();
  throw;  // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
}

/*!\rst
  Register an exception translator (C++ to Python) with boost python for ExceptionType, using the callable translate.
  Boost python expects only unary exception translators (w/the exception to translate as the argument), so we use
  a lambda to capture additional arguments for our translators.

  .. Note:: if/when template'd lambdas become available (C++14?), we can kill this function. It is just a simple
    template'd wrapper around a lambda-expression.

  TEMPLATE PARAMETERS:

  * ExceptionType: the type of the exception that the user wants to register
  * Translator: a Copyconstructible type such that the following code is well-formed::

      void SomeFunc(ExceptionType except, const PyExceptionClassContainer& py_exception_type_objects) {
        translate(except, py_exception_type_objects);
      }

    Currently, the use cases in BOOST_PYTHON_MODULE(GPP) pass translate as a function pointer.
    This follows the requirements for boost::python::register_exception_translator:
    http://www.boost.org/doc/libs/1_55_0/libs/python/doc/v2/exception_translator.html
    http://en.cppreference.com/w/cpp/concept/CopyConstructible

  \param
    :py_exception_type_objects: PyExceptionClassContainer with the appropriate type object for constructing the
      translated Python exception base class. We assume that this type inherits from Exception.
    :translate: an instance of Translator satisfying the requirements described above in TEMPLATE PARAMETERS.
\endrst*/
template <typename ExceptionType, typename Translator>
void RegisterExceptionTranslatorWithPayload(const PyExceptionClassContainer& py_exception_type_objects, Translator translate) {
  static_assert(std::is_copy_constructible<Translator>::value, "Exception translator must be copy constructible.");
  // lambda capturing the closure of translate (py_exception_type_objects)
  // py_exception_type_objects captured by value; we don't want a dangling reference
  // translate captured by value; we know it is copyconstructible
  auto translate_exception =
      [py_exception_type_objects, translate](const ExceptionType& except) {
    translate(except, py_exception_type_objects);
  };

  // nullptr suppresses a superfluous compiler warning b/c boost::python::register_exception_translator
  // defaults a (dummy) pointer argument to 0.
  boost::python::register_exception_translator<ExceptionType>(translate_exception, nullptr);
}

/*!\rst
  Helper that registers exception translators to convert C++ exceptions to Python exceptions.
  This is just a convenient place to register translators for optimal_learning's exceptions.

  .. NOTE:: PyExceptionClassContainer (monostate class) must be properly initialized first!
    Otherwise all translators will translate to ``PyExceptionClassContainer::default_exception_type_object_``
    (e.g., ``PyExc_RuntimeError``).
\endrst*/
void RegisterOptimalLearningExceptions() {
  PyExceptionClassContainer py_exception_type_objects;

  // Note: boost python stores exception translators in a LIFO stack. The most recently (in code execution order)
  // registered translator gets "first shot" at matching incoming exceptions. Reference:
  // http://www.boost.org/doc/libs/1_55_0/libs/python/doc/v2/exception_translator.html
  // TranslateStdException MUST appear first! Otherwise it will *mask* other translate preceding it.
  RegisterExceptionTranslatorWithPayload<std::exception>(py_exception_type_objects, &TranslateStdException);
  RegisterExceptionTranslatorWithPayload<SingularMatrixException>(py_exception_type_objects, &TranslateSingularMatrixException);
  RegisterExceptionTranslatorWithPayload<InvalidValueException<int>>(py_exception_type_objects, &TranslateInvalidValueException<int>);
  RegisterExceptionTranslatorWithPayload<InvalidValueException<double>>(py_exception_type_objects, &TranslateInvalidValueException<double>);
  RegisterExceptionTranslatorWithPayload<BoundsException<int>>(py_exception_type_objects, &TranslateBoundsException<int>);
  RegisterExceptionTranslatorWithPayload<BoundsException<double>>(py_exception_type_objects, &TranslateBoundsException<double>);
}

}  // end unnamed namespace

namespace {  // unnamed namespace for BOOST_PYTHON_MODULE(GPP) definition

// TODO(GH-140): improve docstrings for the GPP module and for the classes, functions, etc
//   in it exposed to Python. Many of them are a bit barebones at the moment.
BOOST_PYTHON_MODULE(GPP) {
  boost::python::scope current_scope;

  // initialize PyExceptionClassContainer monostate class and set its scope to this module (GPP)
  PyExceptionClassContainer py_exception_type_objects;
  py_exception_type_objects.Initialize(&current_scope);

  // Register exception translators to convert C++ exceptions to python exceptions.
  // Note: if adding additional translators, recall that boost python maintains translators in a LIFO
  // stack. See RegisterOptimalLearningExceptions() docs for more details.
  RegisterOptimalLearningExceptions();

  bool show_user_defined = true;
  bool show_py_signatures = true;
  bool show_cpp_signatures = true;
  // enable full docstrings for the functions, enums, ctors, etc. provided in this module.
  boost::python::docstring_options doc_options(show_user_defined, show_py_signatures, show_cpp_signatures);

  current_scope.attr("__doc__") = R"%%(
    This module is the python interface to the C++ component of the Metrics Optimization Engine, or MOE. It exposes
    enumerated types for specifying interface behaviors, C++ objects for computation and communication, as well as
    various functions for optimization and testing/data exploration.

    **OVERVIEW**

    TODO(GH-25): when we come up with a "README" type overview for MOE, that or parts of that should be incorporated here.

    MOE is a black-box global optimization method for objectives (e.g., click-through rate, delivery time, happiness)
    that are time-consuming/expensive to measure, highly complex, non-convex, nontrivial to predict, or all of the above.
    It optimizes the user-chosen objective with respect to the user-specified parameters; e.g., scoring weights,
    thresholds, learning rates, etc.

    MOE examines the history of all experiments (tested parameters, measured metrics, measurement noise) to date and
    outputs the next set of parameters to sample that it believes will produce the best results.

    To perform this optimization, MOE models the world with a Gaussian Process (GP). Conceptually, this means that MOE
    assumes the performance of every set of parameters (e.g., weights) is governed by a Gaussian with some mean and
    variance. The mean/variance are functions of the historical data. Compared to running live experiments,
    computations on a GP are cheap. So MOE uses GPs to predict and maximize the Expected Improvement (EI), producing
    a new set of experiment parameters that produces the greatest (expected) improvement over the best performing
    parameters from the historical data.

    Thus, using MOE breaks down into three main steps:

    1. Model Selection/Hyperparameter Optimization: the GP (through a "covariance" function) has several hyperparameters
       that are not informed by the model. We first need to choose an appropriate set of hyperparameters. For example,
       we could choose the hyperparameters that maximize the likelihood that the model produced the observed data.
       multistart_hyperparameter_optimization() is the primary endpoint for this functionality.

    2. Construct the Gaussian Process: from the historical data and the hyperparameters chosen in step 1, we can build
       a GP that MOE will use as a proxy for the behavior of the objective in the real world. In this sense, the GP's
       predictions are a type of regression. GaussianProcess() constructs a GP.

    3. Select new Experiment Parameters via EI: with the GP from step 2, MOE now has a model for the "real world." Using this
       GP model, we will select the next experiment parameters (or set of parameters) for live measurement. These
       new parameters are the ones MOE thinks will produce the biggest gain over the historical best.
       multistart_expected_improvement_optimization() is the primary endpoint for this functionality.
       heuristic_expected_improvement_optimization() is an alternative endpoint (faster, less accurate).

    Users may specify "p" (aka num_being_sampled) points from ongoing/incomplete experiments for MOE's optimizer to consider.
    And they may request that MOE produce "q" (aka num_to_sample) points representing the parameters for new
    experiments. These are found by solving the q,p-EI problem (see gpp_math.hpp file overview for further details).

    **DETAILS**

    For more information, consult the docstrings for the entities exposed in this module.  (Everybody has one!)
    These docstrings currently provide fairly high level (and sometimes sparse) descriptions.
    For further details, see the file documents for the C++ hpp and cpp files. Header (hpp) files contain more high
    level descriptions/motivation whereas source (cpp) files contain more [mathematical] details.
    gpp_math.hpp and gpp_model_selection.hpp are good starting points for more reading.

    TODO(GH-25): when we have jemdoc (or whatever tool), point this to those docs as well.

    Now we will provide an overview of the enums, classes, and endpoints provided in this module.

    .. Note:: Each entity provided in this module has a docstring; this is only meant to be an overview. Consult the
      individual docstrings for more information and/or see the C++ docs.

    * Exceptions:
      We expose equivalent Python definitions for the exception classes in gpp_exception.hpp. We also provide
      translators so that C++ exceptions will be caught and rethrown as their Python counterparts. The type objects
      (e.g., BoundsException) are referenced by module-scope names in Python.

    * Enums:
      We provide some enum types defined in C++. Values from these enum types are used to signal information about
      which domain, which optimizer, which log likelihood objective, etc. to use throughout this module. We define the
      enums in C++ because not all currently in-use version of Python support enums natively. Additionally, we wanted
      the strong typing. In particular, a function expecting a DomainTypes enum *cannot* take (int) 0 as an input
      even if kTensorProduct maps to the value 0. In general, *never* rely on the particular value of each enum name.

      * DomainTypes
      * LogLikelihoodTypes
      * OptimizerTypes

    * Objects:
      We currently expose constructors for a few C++ objects:

      * GaussianProcess: for one set of historical data (and hyperparameters), this represents the GP model. It is fairly
        expensive to create, so the intention is that users create it once and pass it to all functions in this module
        that need it (as opposed to recreating it every time).  Constructing the GP is noted as step 2 of MOE above.
      * GradientDescentParameters, NewtonParameters: structs that hold tolerances, max step counts, learning rates, etc.
        that control the behavior of the derivative-based optimizers
      * RandomnessSourceContainer: container for a uniform RNG and a normal (gaussian) RNG. These are needed by the C++ to
        guarantee that multi-threaded runs see different (and consistent) randomness. This class also exposes several
        functions for setting thread-safe seeds (both explicitly and automatically).

    * Optimization:
      We expose two main optimization routines. One for model selection and the other for experimental cohort selection.
      These routines provide multistart optimization (with choosable optimizer, domain, and objective type) and are
      the endpoints to use for steps 1 and 3 of MOE dicussed above.
      These routines are multithreaded.

      * multistart_hyperparameter_optimization: optimize the specified log likelihood measure to yield the hyperparameters
        that produce the "best" model.
      * multistart_expected_improvement_optimization: optimize the expected improvement to yield the experimental parameters
        that the user should test next (solving q,p-EI).

    * Testing/Exploring:
      These endpoints are provided for testing core C++ functionality as well as exploring the behaviors of
      Gaussian Processes, Expected Improvement, etc.  GP and EI functions require a GaussianProcess object.

      * Log Likelihood (model fit): compute_log_likelihood, compute_hyperparameter_grad_log_likelihood
      * GPs: get_mean, get_grad_mean, get_var, get_chol_var, get_grad_var, get_grad_chol_var
      * EI: compute_expected_improvement, compute_grad_expected_improvement

    * Plotting:
      These endpoints are useful for plotting log likelihood or expected improvement (or other applications needing
      lists of function values).
      These routines are multithreaded.

      * evaluate_log_likelihood_at_hyperparameter_list: compute selected log likelihood measures for each set of specified
        hyperparameters. Equivalent to but much faster than calling compute_log_likelihood() in a loop.
      * evaluate_EI_at_point_list: compute expected improvement for each point (cohort parameters) in an input list.
        Equivalent to but much faster than calling compute_expected_improvement in a loop.
    )%%";

  ExportCppTestFunctions();
  ExportEnumTypes();
  ExportEstimationPolicies();
  ExportExpectedImprovementFunctions();
  ExportGaussianProcessFunctions();
  ExportModelSelectionFunctions();
  ExportOptimizerParameterStructs();
  ExportRandomnessContainer();
}  // end BOOST_PYTHON_MODULE(GPP) definition

}  // end unnamed namespace

}  // end namespace optimal_learning
