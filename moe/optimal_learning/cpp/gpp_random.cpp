/*!
  \file gpp_random.cpp
  \rst
  This file contains definitions of infrequently-used and/or expensive functions declared in gpp_random.hpp.
  The main purpose was to hide details and get things like boost/functional/hash.hpp and <ostream> out of the
  gpp_random header.
\endrst*/

#include "gpp_random.hpp"

#include <sys/time.h>

#include <cmath>

#include <algorithm>
#include <limits>
#include <ostream>  // NOLINT(readability/streams): streams are the only way pull state data out of boost's PRNG engines
#include <vector>

#include <boost/functional/hash.hpp>  // NOLINT(build/include_order)
#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"

namespace optimal_learning {

void UniformRandomGenerator::SetRandomizedSeed(EngineType::result_type base_seed, int thread_id) noexcept {
  struct timeval time;
  gettimeofday(&time, nullptr);  // nominally time since epoch with microsecond resolution

  // NOTE: EngineType::result_type is likely uint32_t or uint64_t (boost docs).
  // And boost::hash_combine has the signature:
  // template<typename T> void hash_combine(size_t & seed, T const& v);
  // These 3 integer types are each type alises for one of: "unsigned int", "long unsigned int",
  // and "long long unsigned int"; it's implementation dependent.

  // Thus the type of size_t may not be the same (maybe not even the same width!)
  // as EngineType::result_type.  This matters b/c hash_combine() expects a *reference*
  // for its first argument, so we must cast explicitly here.
  std::size_t seed = base_seed;

  // hash_combine is used to create a hash value from several variables
  // use it repeatedly to combine time and thread_id info into the base seed
  boost::hash_combine(seed, time.tv_sec);
  boost::hash_combine(seed, time.tv_usec);
  boost::hash_combine(seed, thread_id);

  // NOTE: emphasizing the previous NOTE, this call discards bits from the final value of seed
  // if sizeof(std::size_t) > sizeof(EngineType::result_type)
  // This behavior is reasonable: the seed value passed to SetExplicitSeed()
  // is the same as long as sizeof(std::size_t) >= sizeof(EngineType::result_type)
  SetExplicitSeed(seed);
}

void UniformRandomGenerator::PrintState(std::ostream * out_stream) const {
  (*out_stream) << engine << "\n";  // NOLINT(readability/streams): this is the only way pull state data out of boost's PRNG engines
}

void NormalRNG::PrintState(std::ostream * out_stream) const {
  (*out_stream) << engine << "\n";  // NOLINT(readability/streams): this is the only way pull state data out of boost's PRNG engines
}

/*!\rst
  domain specifies a domain from which to draw points at uniformly at random; it is a bounding box specification in
  dim pairs of (domain_min, domain_max) values, defining edge-lengths of the hypercube domain.

  Each edge is divied into num_samples equally sized portions.  In 2D, this would be like dividing the domain
  into a (possibly scaled) chessboard.

  Then we visit each dimension of the domain in sequence.  The divisions of the current edge divide the domain into slices
  (e.g., rows/columns in 2D).  Each division is chosen uniformly at random and a random point is placed inside.

  In this way, each "row" and "column" have exactly 1 point in them.  In 2D, if each point is a rook,
  then no rook attacks any other rook.
\endrst*/
void ComputeLatinHypercubePointsInDomain(ClosedInterval const * restrict domain, int dim, int num_samples, UniformRandomGenerator * uniform_generator, double * restrict random_points) {
  std::vector<int> index_array(num_samples);

  for (int i = 0; i < dim; ++i) {
    // partition size in hypercube
    double subcube_edge_length = (domain[i].Length())/static_cast<double>(num_samples);

    // generate a uniform random ordering of sample indexes
    for (int j = 0; j < static_cast<int>(index_array.size()); ++j) {
      index_array[j] = j;
    }
    std::shuffle(index_array.begin(), index_array.end(), uniform_generator->engine);

    boost::uniform_real<double> uniform_double_distribution(0.0, subcube_edge_length);
    for (int j = 0; j < num_samples; ++j) {
      double point_base = domain[i].min + subcube_edge_length*index_array[j];
      random_points[j*dim] = point_base + uniform_double_distribution(uniform_generator->engine);
    }

    random_points += 1;
  }
}

/*!\rst
  We need to draw a set of points, ``x_i``, (uniformly distributed by volume) such that ``x_i >= 0 \forall i`` and
  ``\sum_i x_i <= 1``, also implying that ``x_i <= 1 \forall i``.
  This is precisely the output of the Dirichlet Distribution.  It already has the correct support (domain), and then sampling
  from the unit d-simplex is just a matter of selecting the right parameters for Dirichlet; this turns out to reduce it to
  sampling from an exponential distribution.
  http://en.wikipedia.org/wiki/User:Skinnerd/Simplex_Point_Picking
\endrst*/
void ComputeUniformPointsInUnitSimplex(int dim, int num_samples, UniformRandomGenerator * uniform_generator, double * restrict random_points) {
  boost::uniform_real<double> uniform_double_distribution(std::numeric_limits<double>::min(), 1.0);  // draw from open interval (0,1)
  for (int i = 0; i < num_samples; ++i) {
    // draw from the exponential distribution dim times: generate y_i = uniform in (0,1), then x_i = -ln(y_i)
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) {
      random_points[d] = uniform_double_distribution(uniform_generator->engine);
      random_points[d] = -std::log(random_points[d]);
      sum += random_points[d];
    }
    sum = 1.0/sum;  // cannot divide by 0 b/c random_points[d] > 0 for all d
    VectorScale(dim, sum, random_points);
    random_points += dim;
  }
}

}  // end namespace optimal_learning
