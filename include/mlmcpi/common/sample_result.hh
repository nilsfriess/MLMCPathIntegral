#pragma once

#include <vector>

namespace mlmcpi {
template <typename PathType> struct sample_result {
  std::vector<PathType> samples;
  double acceptance_rate;
};

} // namespace mlmcpi
