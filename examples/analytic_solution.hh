#pragma once

#include <cmath>

inline double analytic_solution(double delta_t, double m0, double mu2,
                                std::size_t path_length) {
  const double R =
      1. + (delta_t * delta_t * mu2) / (2 * m0) -
      delta_t * std::sqrt(mu2) * std::sqrt(1 / m0 + (delta_t * delta_t * mu2) / (4 * m0));
  return 1. / (2. * std::sqrt(mu2) * std::sqrt(m0 + 0.25 * delta_t * delta_t * mu2)) *
         (1. + std::pow(R, path_length)) / (1. - std::pow(R, path_length));
}
