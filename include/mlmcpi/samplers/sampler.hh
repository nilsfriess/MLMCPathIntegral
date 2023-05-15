#pragma once

#include <optional>

namespace mlmcpi {
template <typename Action> struct sampler {
  using PathType = typename Action::PathType;

  virtual std::optional<PathType> perform_step(const PathType &) = 0;

  virtual ~sampler() = default;
};

} // namespace mlmcpi
