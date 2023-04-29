#pragma once

#include <concepts>
#include <type_traits>

namespace mlmcpi {

// clang-format off
  template <typename T, typename DataT = double>
  concept ProposalDistribution =
    std::is_default_constructible_v<T> and requires(T dist) {
    { dist.sample(DataT{}) } -> std::same_as<DataT>;
    { dist.evaluate(DataT{}) } -> std::same_as<double>;
  };

  template <typename T, typename DataT = double>
  concept TargetDistribution =
    std::is_default_constructible_v<T> and requires(T dist) {
    { dist.evaluate(DataT{}) } -> std::same_as<double>;
  };
// clang-format on
} // namespace mlmcpi
