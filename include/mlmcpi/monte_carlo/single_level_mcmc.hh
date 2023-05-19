#pragma once

#include "mlmcpi/common/mcmc_result.hh"
#include "mlmcpi/common/sample_result.hh"
#include "mlmcpi/qoi/identity.hh"

#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace mlmcpi {

template <typename Sampler> struct single_level_mcmc {
  using PathType = typename Sampler::PathType;

  single_level_mcmc(Sampler &sampler_) : sampler{sampler_} {}

  template <typename QOI = mlmcpi::identity<PathType>>
  mcmc_result<typename QOI::ResultType> run(std::size_t n_burnin, std::size_t n_samples,
                                            PathType initial_path) {
    QOI qoi;
    mcmc_result<typename QOI::ResultType> result{n_samples};

    auto current = initial_path;
    for (std::size_t i = 0; i < n_burnin; ++i) {
      const auto proposal = sampler.perform_step(current);
      current = proposal.value_or(current);
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
      const auto proposal = sampler.perform_step(current);
      current = proposal.value_or(current);

      result.add_sample(qoi(std::forward<PathType>(current)), proposal.has_value());
    }

    return result;
  }

private:
  Sampler &sampler;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
