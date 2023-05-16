#pragma once

#include <blaze/Blaze.h>
#include <memory>

namespace mlmcpi {

struct harmonic_oscillator_action {
  using PathType = blaze::DynamicVector<double>;

  harmonic_oscillator_action(double delta_t_) noexcept
      : delta_t(delta_t_),
        W_curvature_((2. / delta_t + delta_t * mu2) * m0),
        W_minimum_scaling(0.5 / (1. + 0.5 * delta_t * delta_t * mu2)) {}

  double evaluate(const PathType &path) const {
    // First term is computed separately using periodic BCs
    auto dxdt = (path[0] - path[path.size() - 1]) / delta_t;
    auto dxdt2 = dxdt * dxdt;
    double res = (dxdt2 + mu2 * path[0] * path[0]);

    for (std::size_t i = 1; i < path.size(); ++i) {
      dxdt = (path[i] - path[i - 1]) / delta_t;
      dxdt2 = dxdt * dxdt;
      res += dxdt2 + mu2 * path[i] * path[i];
    }
    return 0.5 * m0 * delta_t * res;
  }

  PathType evaluate_force(const PathType &path) const {
    PathType force(path.size());

    double A = m0 / delta_t;
    double B = 2. + delta_t * delta_t * mu2;

    force[0] = A * (B * path[0] - path[path.size() - 1] - path[1]);

    for (std::size_t i = 1; i < path.size() - 1; ++i)
      force[i] = A * (B * path[i] - path[i - 1] - path[i + 1]);

    force[path.size() - 1] =
        A * (B * path[path.size() - 1] - path[path.size() - 2] - path[0]);

    return force;
  }

  double analytic_solution(std::size_t path_length) const {
    double R = 1. + 0.5 * delta_t * delta_t * mu2 -
               delta_t * std::sqrt(mu2) * std::sqrt(1. + 0.25 * delta_t * delta_t * mu2);
    return 1. /
           (2. * m0 * std::sqrt(mu2) * std::sqrt(1 + 0.25 * delta_t * delta_t * mu2)) *
           (1. + std::pow(R, path_length)) / (1. - std::pow(R, path_length));
  }

  inline double W_curvature(double /*x_m*/, double /*x_p*/) const { return W_curvature_; }
  inline double W_minimum(double x_m, double x_p) const {
    return W_minimum_scaling * (x_m + x_p);
  }

  harmonic_oscillator_action make_coarsened_action(double factor = 2) const {
    auto coarse_action(*this);
    coarse_action.delta_t *= factor;
    return coarse_action;
  }

private:
  double delta_t;

  double W_curvature_;
  double W_minimum_scaling;

  constexpr static double m0 = 1;
  constexpr static double mu2 = 1;
};

} // namespace mlmcpi
