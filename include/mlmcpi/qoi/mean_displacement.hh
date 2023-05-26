#pragma once

#include "mlmcpi/common/math.hh"

namespace mlmcpi {
template <typename PathType, typename DataT = double> struct mean_displacement {
  using ResultType = DataT;

  ResultType operator()(const PathType &path) { return mean(path * path); }
};
} // namespace mlmcpi
