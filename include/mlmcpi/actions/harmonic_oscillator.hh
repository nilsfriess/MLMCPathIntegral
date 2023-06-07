#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

namespace mlmcpi {

template <typename TPathType> struct harmonic_oscillator_action {
  using PathType = TPathType;

  harmonic_oscillator_action(std::size_t path_length_, double delta_t_, double m0_ = 1.,
                             double mu2_ = 1.) noexcept
      : path_length{path_length_},
        delta_t{delta_t_},
        m0{m0_},
        mu2{mu2_},
        W_curvature_((2. / delta_t + delta_t * mu2) * m0),
        W_minimum_scaling(0.5 / (1. + 0.5 * delta_t * delta_t * mu2)) {}

  double evaluate(const PathType &path) const {
    assert(path.size() == path_length);

    // First term is computed separately using periodic BCs
    auto dxdt = (path[0] - path[path.size() - 1]) / delta_t;
    auto dxdt2 = dxdt * dxdt;
    double res = m0 * dxdt2 + mu2 * path[0] * path[0];

    for (std::size_t i = 1; i < path.size(); ++i) {
      dxdt = (path[i] - path[i - 1]) / delta_t;
      dxdt2 = dxdt * dxdt;
      res += m0 * dxdt2 + mu2 * path[i] * path[i];
    }

    return 0.5 * delta_t * res;
  }

  PathType grad_potential(const PathType &path) const {
    assert(path.size() == path_length);

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

  inline double W_curvature(double /*x_m*/, double /*x_p*/) const { return W_curvature_; }
  inline double W_minimum(double x_m, double x_p) const {
    return W_minimum_scaling * (x_m + x_p);
  }

  harmonic_oscillator_action make_coarsened_action() const {
    auto coarse_action(*this);
    coarse_action.delta_t *= 2;
    coarse_action.path_length /= 2;
    return coarse_action;
  }

  harmonic_oscillator_action make_finer_action() const {
    auto coarse_action(*this);
    coarse_action.delta_t /= 2;
    coarse_action.path_length *= 2;
    return coarse_action;
  }

  std::size_t get_path_length() const { return path_length; }

private:
  std::size_t path_length;
  double delta_t;

  double m0;
  double mu2;

  double W_curvature_;
  double W_minimum_scaling;
};

} // namespace mlmcpi
