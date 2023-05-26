#pragma once

#ifdef USE_BLAZE
#include <blaze/Blaze.h>
#else
#include <limits>
#endif

namespace mlmcpi {
template <typename Vector> inline double sqrNorm([[maybe_unused]] const Vector &vec) {
#ifdef USE_BLAZE
  return blaze::sqrNorm(vec);
#else
  return std::numeric_limits<double>::infinity();
#endif
}

template <typename Vector> inline double mean([[maybe_unused]] const Vector &vec) {
#ifdef USE_BLAZE
  return blaze::mean(vec);
#else
  return std::numeric_limits<double>::infinity();
#endif
}

}; // namespace mlmcpi
