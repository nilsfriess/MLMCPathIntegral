#pragma once

#include "mlmcpi/common/mcmc_result.hh"
#include "mlmcpi/common/sample_result.hh"
#include "mlmcpi/qoi/identity.hh"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace mlmcpi {

template <typename Sampler> struct single_level_mcmc {
  using PathType = typename Sampler::PathType;

  single_level_mcmc(Sampler &sampler_) : sampler{sampler_} {}

  template <typename QOI = mlmcpi::identity<PathType>>
  mcmc_result<typename QOI::ResultType> run(std::size_t n_burnin, PathType initial_path,
                                            double target_error = 1e-2,
                                            std::size_t max_steps = 10000) {
    QOI qoi;
    mcmc_result<typename QOI::ResultType> result;

    auto current = initial_path;
    for (std::size_t i = 0; i < n_burnin; ++i) {
      const auto proposal = sampler.perform_step(current);
      current = proposal.value_or(current);
    }

    const auto compute_required_samples = [&]() {
      const auto autocorr_time = result.integrated_autocorr_time();
      const auto var = result.variance();

      if (autocorr_time == 0)
        return std::numeric_limits<std::size_t>::max();

      return static_cast<std::size_t>(
          std::ceil((autocorr_time * var) / (target_error * target_error)));
    };

    std::size_t step = 1;
    std::size_t required_samples = std::numeric_limits<std::size_t>::max();

    while (step <= required_samples && step <= max_steps) {
      const auto proposal = sampler.perform_step(current);
      current = proposal.value_or(current);

      result.add_sample(qoi(std::forward<PathType>(current)), proposal.has_value());

      // Check if we have enough samples for the required error every 100 steps
      if (step % 100 == 0) {
        required_samples = compute_required_samples();

        if (result.mean_error() < 1e-12)
          required_samples = std::numeric_limits<std::size_t>::max();
      }

      step++;
    }

    return result;
  }

private:
  Sampler &sampler;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
