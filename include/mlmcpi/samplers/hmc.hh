#pragma once

#include "mlmcpi/samplers/sampler.hh"

#include <blaze/Blaze.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

namespace mlmcpi {
template <typename Action, typename Engine = std::mt19937>
struct hmc_sampler : sampler<Action> {
  using PathType = blaze::DynamicVector<double>;

  hmc_sampler(std::shared_ptr<Action> action_, std::shared_ptr<Engine> engine_)
      : action{std::move(action_)}, engine{std::move(engine_)} {}

  std::optional<PathType> perform_step(const PathType &current) override {
    auto position = current;
    // Generate random initial momentum
    PathType momentum(current.size());
    std::generate(momentum.begin(), momentum.end(),
                  [&]() { return normal_dist(*engine); });

    auto initial_kinetic = 0.5 * blaze::sqrNorm(momentum);

    constexpr int timesteps = 100;
    constexpr double dt = 0.12;
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

private:
  std::shared_ptr<Action> action;

  std::shared_ptr<Engine> engine;
  std::normal_distribution<double> normal_dist;

  std::uniform_real_distribution<double> unif_dist;
};
} // namespace mlmcpi
