#pragma once

#include <blaze/math/expressions/Forward.h>

namespace mlmcpi {

template <typename PathType, typename DataT = double> struct mean_displacement {
  using ResultType = DataT;

  ResultType operator()(const PathType &path) {
    return blaze::mean(path * path);
  }
};
} // namespace mlmcpi
