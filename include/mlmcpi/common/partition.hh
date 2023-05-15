#pragma once

#include <algorithm>
#include <tuple>

namespace mlmcpi {
template <typename PathType>
inline std::tuple<PathType, PathType> partition_odd_even(const PathType &path) {
  PathType odd(path.size() / 2);
  PathType even(path.size() / 2);

  bool is_even = true;
  for (std::size_t i = 0; i < path.size(); ++i) {
    if (is_even)
      even[i / 2] = path[i];
    else
      odd[(i - 1) / 2] = path[i];

    is_even = not is_even;
  }
  return {odd, even};
}

template <typename PathType>
PathType combine_odd_even(const PathType &odd, const PathType &even) {
  PathType res(even.size() + odd.size());

  for (std::size_t i = 0; i < res.size(); ++i) {
    if (i % 2 == 0)
      res[i] = even[i / 2];
    else
      res[i] = odd[(i - 1) / 2];
  }

  return res;
}

} // namespace mlmcpi
