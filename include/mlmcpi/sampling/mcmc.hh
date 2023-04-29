#pragma once

#include "concepts.hh"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace mlmcpi {

template <typename DataT, TargetDistribution<DataT> P,
          ProposalDistribution<DataT> Q>
struct mcmc {

  std::vector<DataT> sample(std::size_t n_burnin, std::size_t n_samples,
                            DataT initial_sample = DataT{0}) {
    std::vector<DataT> samples(n_burnin + n_samples);
    samples[0] = initial_sample;

    P target_dist;
    Q proposal_dist;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> unif_dist;

    for (std::size_t i = 1; i < n_burnin + n_samples; ++i) {
      const auto current = samples.at(i - 1);
      const auto proposal = proposal_dist.sample(current);

      const auto acceptance_ratio =
          target_dist.evaluate(proposal) / target_dist.evaluate(current);
      const auto acceptance_prob = std::min(1., acceptance_ratio);

      if (unif_dist(generator) <= acceptance_prob)
        samples.at(i) = proposal;
      else
        samples.at(i) = current;
    }

    samples.erase(samples.begin(), samples.begin() + n_burnin);
    return samples;
  }
};
} // namespace mlmcpi
