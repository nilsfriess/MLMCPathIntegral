#pragma once

#include "mlmcpi/samplers/sampler.hh"

#include <blaze/Blaze.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

namespace mlmcpi {
template <typename Action, typename Engine = std::mt19937>
struct hmc_sampler : sampler<Action> {
  using PathType = blaze::DynamicVector<double>;

  hmc_sampler(double stepsize, std::shared_ptr<Action> action_,
              std::shared_ptr<Engine> engine_)
      : dt{stepsize}, action{std::move(action_)}, engine{std::move(engine_)} {}

  std::optional<PathType> perform_step(const PathType &current) override {
    auto position = current;
    // Generate random initial momentum
    PathType momentum(current.size());
    std::generate(momentum.begin(), momentum.end(),
                  [&]() { return normal_dist(*engine); });

    auto initial_kinetic = 0.5 * blaze::sqrNorm(momentum);

    constexpr int timesteps = 100;
    for (int k = 0; k < timesteps; ++k) {
      auto dt_momentum = dt;
      auto dt_position = dt;

      if (k == 0)
        dt_momentum = 0.5 * dt;
      if (k == timesteps - 1) {
        dt_momentum = 0.5 * dt;
        dt_position = 0;
      }

      auto force = action->evaluate_force(position);
      momentum -= dt_momentum * force;
      position += dt_position * momentum;
    }

    auto final_kinetic = 0.5 * blaze::sqrNorm(momentum);

    const auto delta_S = action->evaluate(position) - action->evaluate(current);
    const auto delta_T = final_kinetic - initial_kinetic;
    const auto delta_H = delta_S + delta_T;

    if (delta_H < 0)
      return position;

    auto acceptance_prob = std::exp(-delta_H);
    if (unif_dist(*engine) < acceptance_prob)
      return position;
    else
      return {};
  }

  std::optional<double> autotune_stepsize(const PathType &initial_path,
                                          double acceptance_rate_target = 0.8) {
    const std::size_t n_samples = 1000;
    const std::size_t n_repetitions = 100;
    const double dt_initial = dt;

    double dt_min = 0.1 * dt;
    double dt_max = 10 * dt;

    auto current = initial_path;

    // Perform burnin
    std::size_t n_burnin = 1000;
    for (std::size_t i = 0; i < n_burnin; ++i) {
      auto proposal = perform_step(current);
      current = proposal.value_or(current);
    }

    for (std::size_t run = 0; run < n_repetitions; ++run) {
      std::size_t accepted_samples = 0;

      dt = 0.5 * (dt_min + dt_max);
      for (std::size_t i = 0; i < n_samples; ++i) {
        auto proposal = perform_step(current);
        if (proposal) {
          current = proposal.value();
          accepted_samples++;
        }
      }

      const auto acceptance_rate = (1. * accepted_samples) / n_samples;
      if (acceptance_rate > acceptance_rate_target)
        dt_min = dt;
      else
        dt_max = dt;

      if (std::abs(acceptance_rate - acceptance_rate_target) < 1e-2)
        return dt;
    }

    dt = dt_initial;
    return {};
  }

private:
  double dt;

  std::shared_ptr<Action> action;

  std::shared_ptr<Engine> engine;
  std::normal_distribution<double> normal_dist;

  std::uniform_real_distribution<double> unif_dist;
};
} // namespace mlmcpi
