#pragma once

#include <cmath>
#include <memory>

namespace mlmcpi {

template <typename Action, typename ProposalDistribution>
struct default_log_acceptance_probability {
  using PathType = typename Action::PathType;

  default_log_acceptance_probability(
      std::shared_ptr<Action> action,
      std::shared_ptr<ProposalDistribution> proposal_dist)
      : action{std::move(action)}, proposal_dist{std::move(proposal_dist)} {}

  double operator()(const PathType &current, const PathType &proposal) {
    return action->evaluate(proposal) - action->evaluate(current) +
           std::log(proposal_dist->density(proposal, current)) -
           std::log(proposal_dist->density(current, proposal));
  }

private:
  std::shared_ptr<Action> action;
  std::shared_ptr<ProposalDistribution> proposal_dist;
};

} // namespace mlmcpi
