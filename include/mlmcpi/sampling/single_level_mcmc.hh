#pragma once

#include "mlmcpi/types.hh"

#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace mlmcpi {

template <typename Action, typename ProposalDist>

struct single_level_mcmc {
  using PathType = typename Action::PathType;

  single_level_mcmc(std::shared_ptr<Action> action,
                    std::shared_ptr<ProposalDist> proposal_dist)
      : action{std::move(action)}, proposal_distribution{
                                       std::move(proposal_dist)} {}

  std::vector<PathType> sample(std::size_t n_burnin, std::size_t n_samples,
                               PathType initial_path) {
    std::vector<PathType> samples;
    samples.reserve(n_burnin + n_samples);
    samples.push_back(initial_path);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> unif_dist;

    for (std::size_t i = 1; i < n_burnin + n_samples; ++i) {
      const auto current = samples.at(i - 1);
      const auto proposal = proposal_distribution->sample(current);

      auto delta_S =
          action->evaluate(proposal) - action->evaluate(current) +
          std::log(proposal_distribution->density(proposal, current)) -
          std::log(proposal_distribution->density(current, proposal));

      if (delta_S < 0) {
        samples.push_back(proposal);
      } else {
        auto acceptance_ratio = std::exp(-delta_S);
        if (unif_dist(generator) < acceptance_ratio)
          samples.push_back(proposal);
        else
          samples.push_back(current);
      }
    }

    samples.erase(samples.begin(), samples.begin() + n_burnin);
    return samples;
  }

private:
  std::shared_ptr<Action> action;
  std::shared_ptr<ProposalDist> proposal_distribution;
};

} // namespace mlmcpi
