#pragma once

#include "mlmcpi/common/partition.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace mlmcpi {
template <typename Action, typename Engine = std::mt19937>
class gaussian_even_odd_conditional {
public:
  using PathType = typename Action::PathType;

  gaussian_even_odd_conditional(const Action &action_, Engine &engine_)
      : action{action_},
        engine{engine_} {}

  PathType sample(const PathType &even_points) {
    assert(2 * even_points.size() == action.get_path_length());

    PathType odd_points(even_points.size());

    for (std::size_t i = 0; i < even_points.size() - 1; ++i) {
      const auto x_m = even_points[i];
      const auto x_p = even_points[i + 1];
      const auto x_min = action.W_minimum(x_m, x_p);
      const auto sigma = 1. / std::sqrt(action.W_curvature(x_m, x_p));
      odd_points[i] = x_min + normal_dist(engine) * sigma;
    }

    // Treat final point using periodic BC's
    const auto x_m = even_points[even_points.size() - 1];
    const auto x_p = even_points[0];
    const auto x_min = action.W_minimum(x_m, x_p);
    const auto sigma = 1. / std::sqrt(action.W_curvature(x_m, x_p));

    odd_points[even_points.size() - 1] = x_min + normal_dist(engine) * sigma;

    return odd_points;
  }

  double log_density(const PathType &path) const {
    assert(path.size() == action.get_path_length());
    const auto size = path.size();

    auto x_m = path[size - 2];
    auto x_p = path[0];
    auto dx = path[size - 1] - action.W_minimum(x_m, x_p);
    auto curvature = action.W_curvature(x_m, x_p);

    double S = 0.5 * curvature * dx * dx - 0.5 * std::log(curvature);
    for (std::size_t i = 0; i < size / 2 - 1; ++i) {
      x_m = path[2 * i];
      x_p = path[2 * (i + 1)];
      dx = path[2 * i + 1] - action.W_minimum(x_m, x_p);
      curvature = action.W_curvature(x_m, x_p);
      S += 0.5 * curvature * dx * dx - 0.5 * std::log(curvature);
    }

    return S;
  }

private:
  const Action &action;
  Engine &engine;

  std::normal_distribution<double> normal_dist;
};

} // namespace mlmcpi
