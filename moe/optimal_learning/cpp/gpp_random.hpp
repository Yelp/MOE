/*!
  \file gpp_random.hpp
  \rst
  This file specifies two classes for abstracting/handling psuedo-random number generation.  Currently, we have:

  1. UniformRandomGenerator (container for a PRNG "engine")
  2. NormalRNG (functor for N(0, 1)-distributed PRNs, uses UniformRandomGenerator)

  It additionally contains two methods for randomly generating points in a tensor-product domain:
  ``[x_0_min, x_0_max] X [x_1_min, x_1_max] X ... X [x_d_min, x_d_max]``
  This file supports a naive sampling and a latin hypercube sampling.

  Lastly, this file contains a method for randomly generating points in a unit simplex domain.

  These classes exist to ease the use of PRNGs--in particular, they remember the most recent seed so that "rollbacks"
  are easy; and they make it easy to generate unique seeds in multi-threaded environments.  The seeding allows users to
  specify a seed or have the class generate one using a combination of time-of-day and thread id.  They also provide a method to
  print the underlying PRNG's current state.

  UniformRandomGenerator is currently a wrapper for the ``boost::mt19937`` engine (although any boost, C++11, etc. engine
  will work), which is the mersenne twister using a common set of parameters.  (The same functionality is also available
  through ``C++11``'s ``<random>``.)  Unlike NormalRNG, this class is not a functor since the range and even type can change
  frequently in usage.  Thus the actual distribution should be constructed as needed (e.g., see ComputeLatinHypercubePointsInDomain).

  NormalRNG is a functor for generating ``mean = 0, variance = 1``, normally distributed (pseudo) random numbers.  In addition
  to UniformRandomGenerator, NormalRNG also implements operator() to draw from the aforementioned distribution.  N(0, 1) is
  a common choice (and the only one used in gpp_* so far), so NormalRNG wraps the entire number generation process.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_HPP_

#include <iosfwd>
#include <vector>

#include <boost/random/mersenne_twister.hpp>  // NOLINT(build/include_order)
#include <boost/random/normal_distribution.hpp>  // NOLINT(build/include_order)
#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)
#include <boost/random/variate_generator.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"

namespace optimal_learning {

/*!\rst
  Abstract class for a functor for generating random numbers distributed ~ N(0, 1) (i.e., standard normal).
  This interface currently does not specify many facilities for seeding the underlying RNG as these 
  (particularly variable width) can vary by implementation.

  This class *only* has pure virtual functions.
\endrst*/
class NormalRNGInterface {
 public:
  /*!\rst
    Generate a random number from standard normal distribution.

    \return
      a random number from a standard (``N(0, 1)``) normal distribution
  \endrst*/
  virtual double operator()() = 0;

  /*!\rst
    Reseeds the generator with its most recently specified seed value.
    Useful for testing--e.g., can conduct multiple runs with the same initial conditions
  \endrst*/
  virtual void ResetToMostRecentSeed() noexcept = 0;

  virtual ~NormalRNGInterface() = default;
};

/*!\rst
  Container for an uniform random generator (e.g., mersenne twister).  Member functions are for easy manipulation
  of seeds and have signatures matching corresponding members of NormalRNG.

  .. Note:: seed values take type ``EngineType::result_type``. Do not pass in a wider integer type!

  .. WARNING:: this class is NOT THREAD-SAFE. You must construct one object per thread (and
  ensure that the seeds are different for practical computations).
\endrst*/
struct UniformRandomGenerator final {
  using EngineType = boost::mt19937;

  //! Default seed value to make reproducing test results simple.
  static constexpr EngineType::result_type kDefaultSeed = 314;

  /*!\rst
    Default-constructs a UniformRandomGenerator, seeding with kDefaultSeed.
  \endrst*/
  UniformRandomGenerator() noexcept;

  /*!\rst
    Construct a UniformRandomGenerator, seeding with the specified seed.
    See UniformRandomGenerator::SetExplicitSeed for details.

    \param
      :seed: new seed to set
  \endrst*/
  explicit UniformRandomGenerator(EngineType::result_type seed) noexcept;

  /*!\rst
    Construct a UniformRandomGenerator, seeding with an automatically selected seed based on time, thread_id, etc.
    See UniformRandomGenerator::SetRandomizedSeed for details.

    \param
      :base_seed: base value for the new seed
      :thread_id: id of the thread using this object
  \endrst*/
  UniformRandomGenerator(EngineType::result_type seed, int thread_id) noexcept;

  /*!\rst
    Get a reference to the RNG engine used by this class.

    Not necessary for this class since ``engine`` is public but we expose this to maintain a uniform
    interface with NormalRNG.

    \return
      reference to the underlying RNG engine
  \endrst*/
  EngineType& GetEngine() noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return engine;
  }

  EngineType::result_type last_seed() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return last_seed_;
  }

  /*!\rst
    Seed the random number generator with the input value.
    The main purpose of this function is for testing--to allow seeding the RNG with
    a known value for repeatability.

    \param
      :seed: new seed to set
  \endrst*/
  void SetExplicitSeed(EngineType::result_type seed) noexcept;

  /*!\rst
    Set a new seed for the random number generator.  A "random" seed is selected based on
    the input seed value, the current time, and the thread_id.

    This function is meant to initialize unique UniformRandomGenerator objects for
    each computation thread::

      std::vector<UniformRandomGenerator> uniform_generator_vector(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        uniform_generator_vector.SetRandomizedSeed(base_seed, i);
      }

    This function is meant to generate seeds so that:

    1. this function can be called multiple times successively (e.g., in the above loop)
       with different thread_ids to initialize RNGs for multiple threads
    2. multiple runs of this code are unlikely to generate the same seed values

    Item 2. is important for minimizing the probability that we run EI computations
    (see gpp_math.hpp) with the "same" randomness.

    \param
      :base_seed: base value for the new seed
      :thread_id: id of the thread using this object
  \endrst*/
  void SetRandomizedSeed(EngineType::result_type base_seed, int thread_id) noexcept;

  /*!\rst
    Reseeds the generator with its most recently specified seed value.
    Useful for testing--e.g., can conduct multiple runs with the same initial conditions
  \endrst*/
  void ResetToMostRecentSeed() noexcept;

  /*!\rst
    Prints the state of the generator to specified ostream.  For testing.

    \param
      :out_stream[1]: a ``std::ostream`` object ready for `operator<<`` use
    \output
      :out_stream[1]: ``std::ostream`` with the engine state "written" to its ``operator<<``
  \endrst*/
  void PrintState(std::ostream * out_stream) const OL_NONNULL_POINTERS;

  bool operator==(const UniformRandomGenerator& other) const;

  bool operator!=(const UniformRandomGenerator& other) const;

  //! An (boost) PRNG engine that can be passed to a ``<boost/random>`` distribution, e.g., ``uniform_real<>``.
  EngineType engine;

 private:
  //! The last seed value that was written to ``engine``. Useful for testing.
  EngineType::result_type last_seed_;
};

/*!\rst
  Functor for computing normally distributed (N(0, 1)) random numbers.
  Uses/maintains an uniform RNG (currently UniformRandomGenerator) and transforms the output to be
  distributed ~ N(0, 1).

  .. Note:: seed values take type ``EngineType::result_type``. Do not pass in a wider integer type!

  .. WARNING:: this class is NOT THREAD-SAFE. You must construct one object per thread (and
    ensure that the seeds are different for practical computations).
\endrst*/
class NormalRNG final : public NormalRNGInterface {
 public:
  using UniformGeneratorType = UniformRandomGenerator;
  using EngineType = UniformRandomGenerator::EngineType;

  //! Default seed value to make reproducing test results simple.
  static constexpr EngineType::result_type kDefaultSeed = 314;

  /*!\rst
    Default-constructs a NormalRNG, seeding with kDefaultSeed.
  \endrst*/
  NormalRNG() noexcept;

  /*!\rst
    Construct a NormalRNG, seeding with the specified seed.
    See NormalRNG::SetExplicitSeed for details.

    \param
      :seed: new seed to set
  \endrst*/
  explicit NormalRNG(EngineType::result_type seed) noexcept;

  /*!\rst
    Construct a NormalRNG, seeding with an automatically selected seed based on time, thread_id, etc.
    See NormalRNG::SetRandomizedSeed for details.

    \param
      :base_seed: base value for the new seed
      :thread_id: id of the thread using this object
  \endrst*/
  NormalRNG(EngineType::result_type seed, int thread_id) noexcept;

  /*!\rst
    Get a reference to the RNG engine used by this class.

    \return
      reference to the underlying RNG engine
  \endrst*/
  EngineType& GetEngine() noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return uniform_generator.engine;
  }

  virtual double operator()();

  EngineType::result_type last_seed() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return uniform_generator.last_seed();
  }

  /*!\rst
    Clears state of ``normal_distribution_`` so that future uses do not depend on any previous actions.
    This is important: the underlying normal distribution likely generates numbers \emph{two} at a time.
    So re-seeding the engine WITHOUT resetting can lead to surprising behavior.
  \endrst*/
  void ResetGenerator() noexcept;

  /*!\rst
    Seed the random number generator with the input value.
    See UniformRandomGenerator::SetExplicitSeed() for more information.

    \param
      :seed: new seed to set
  \endrst*/
  void SetExplicitSeed(EngineType::result_type seed) noexcept;

  /*!\rst
    Set a new seed for the random number generator.  A "random" seed is selected based on
    the input seed value, the current time, and the thread_id.
    See UniformRandomGenerator::SetExplicitSeed() for more information.

    \param
      :seed: base value for the new seed
      :thread_id: id of the thread using this object
  \endrst*/
  void SetRandomizedSeed(EngineType::result_type seed, int thread_id) noexcept;

  /*!\rst
    Reseeds the generator with its most recently specified seed value.
    Useful for testing--e.g., can conduct multiple runs with the same initial conditions
  \endrst*/
  virtual void ResetToMostRecentSeed() noexcept;

  /*!\rst
    Prints the state of the generator to specified ostream.  For testing.

    \param
      :out_stream[1]: a std::ostream object ready for operator<< use
    \output
      :out_stream[1]: std::ostream with engine state <<'d to it
  \endrst*/
  void PrintState(std::ostream * out_stream) const OL_NONNULL_POINTERS;

  //! The underlying generator providing uniform PRNGs for this object to transform to N(0, 1).
  UniformGeneratorType uniform_generator;

 private:
  //! Object for ransforming from uniform to N(0, 1); may carry internal state (e.g., normal random numbers generated 2 at a time).
  boost::normal_distribution<double> normal_distribution_;
  //! Object (for convenience) providing operator() that returns a value distributed ~ N(0, 1).
  boost::variate_generator<EngineType&, boost::normal_distribution<double> > normal_random_variable_;
};

/*!\rst
  RNG that generates normally distributed (N(0,1)) random numbers simply by reading random numbers stored in
  its "random_number_table", a data member in this class.

  .. Note:: this class is used in unit test only, and you have to be careful to ensure the total number of random numbers
    generated from last reset must be smaller than size of "random_number_table", otherwise exception will be thrown.

  .. Warning:: this class is NOT THREAD-SAFE. You must construct one object per thread.
\endrst*/
class NormalRNGSimulator final : public NormalRNGInterface {
 public:
  /*!\rst
    Construct a NormalRNGSimulator by providing table storing random numbers, and size of this random table.

    \param
      :random_number_table_in: pointer to the table storing random numbers
      :size_of_table_in: size of the random table
  \endrst*/
  explicit NormalRNGSimulator(const std::vector<double>& random_number_table_in);

  virtual double operator()();

  virtual void ResetToMostRecentSeed() noexcept;

  int index() const {
    return index_;
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(NormalRNGSimulator);

 private:
  //! Table that stores all random numbers.
  std::vector<double> random_number_table_;
  //! Index of the random number in the table to return when the generator is called.
  int index_;
};

/*!\rst
  Computes a set of random points inside some domain that lie in a latin hypercube.  In 2D, a latin hypercube is a latin
  square--a checkerboard--such that there is exactly one sample in each row and each column.  This notion is generalized
  for higher dimensions where each dimensional "slice" has precisely one sample.

  See wikipedia: http://en.wikipedia.org/wiki/Latin_hypercube_sampling
  for more details on the latin hypercube sampling process.

  \param
    :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
    :dim: the number of spatial dimensions
    :num_samples: number of random points desired
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have changed state due to random draws
    :random_points[num_samples][dim]: array containing random points inside the domain
\endrst*/
OL_NONNULL_POINTERS void ComputeLatinHypercubePointsInDomain(ClosedInterval const * restrict domain, int dim, int num_samples, UniformRandomGenerator * uniform_generator, double * restrict random_points);

/*!\rst
  Computes a set of random points that lie inside a dim-dimensional simplex or d-simplex.
  The points are uniformly-distributed by volume.

  \param
    :dim: the number of spatial dimensions
    :num_samples: number of random points desired
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have changed state due to random draws
    :random_points[num_samples][dim]: array containing random points inside the domain
\endrst*/
OL_NONNULL_POINTERS void ComputeUniformPointsInUnitSimplex(int dim, int num_samples, UniformRandomGenerator * uniform_generator, double * restrict random_points);

/*!\rst
  Computes a "random" point guaranteed to be within the specified domain boundaries (inclusive).
  No inputs may be nullptr.

  "random": This code simply draws a uniform random coordinate from this direction.  Each coordinate is drawn
  independently and multiple calls to the function are independent.  As indicated under "random sampling" here,
  http://en.wikipedia.org/wiki/Latin_hypercube_sampling
  the we have no guarantees on the actual distribution of the resulting point set since we make no effort to control it.

  WARNING: this function is NOT THREAD-SAFE.

  \param
    :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
    :dim: the number of spatial dimensions
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :random_point[dim]: array containing a random point inside the domain
\endrst*/
inline OL_NONNULL_POINTERS void ComputeRandomPointInDomain(ClosedInterval const * restrict domain, int dim, UniformRandomGenerator * uniform_generator, double * restrict random_point) noexcept {
  for (int i = 0; i < dim; ++i) {
    boost::uniform_real<double> uniform_double(domain[i].min, domain[i].max);
    random_point[i] = uniform_double(uniform_generator->engine);
  }
}

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_HPP_
