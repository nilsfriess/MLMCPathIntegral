#pragma once

#include <concepts>
#include <type_traits>

namespace mlmcpi {

// clang-format off
  template <typename T>
  concept ProposalDistribution =
    std::is_default_constructible_v<T> and requires(T dist) {
    { dist.sample(0.) } -> std::same_as<double>;
    { dist.evaluate(0.) } -> std::same_as<double>;
  };

  template <typename T>
  concept TargetDistribution =
    std::is_default_constructible_v<T> and requires(T dist) {
    { dist.evaluate(0.) } -> std::same_as<double>;
  };
// clang-format on

} // namespace mlmcpi
