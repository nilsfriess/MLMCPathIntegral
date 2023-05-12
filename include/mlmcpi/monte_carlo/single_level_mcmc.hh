#pragma once

#include "mlmcpi/common/sample_result.hh"

#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace mlmcpi {

template <typename Sampler> struct single_level_mcmc {
  using PathType = typename Sampler::PathType;

  single_level_mcmc(std::shared_ptr<Sampler> sampler)
      : sampler{std::move(sampler)} {}

  sample_result<PathType> sample(std::size_t n_burnin, std::size_t n_samples,
                                 PathType initial_path) {
    std::vector<PathType> samples;
    samples.reserve(n_burnin + n_samples);
    samples.push_back(initial_path);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> unif_dist;

    [[maybe_unused]] std::size_t rejected_samples = 0;

    for (std::size_t i = 1; i < n_burnin + n_samples; ++i) {
      const auto current = samples.at(i - 1);
      const auto proposal = sampler->perform_step(current);

      samples.push_back(proposal.value_or(current));

      if (!proposal && (i > n_burnin))
        rejected_samples++;
    }

    samples.erase(samples.begin(), samples.begin() + n_burnin);

    sample_result<PathType> res;
    res.samples = samples;
    res.acceptance_rate =
        (n_samples - rejected_samples) / static_cast<double>(n_samples);
    return res;
  }

private:
  std::shared_ptr<Sampler> sampler;
};

} // namespace mlmcpi
