#pragma once

#include <utility>

namespace mlmcpi {
template <typename DataT> struct identity {
  using ResultType = DataT;

  constexpr DataT &&operator()(DataT &&t) const { return std::forward<DataT>(t); }
};
} // namespace mlmcpi
