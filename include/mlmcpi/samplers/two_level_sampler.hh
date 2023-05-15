#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <random>

#include "mlmcpi/common/partition.hh"

namespace mlmcpi {
template <typename Action, typename OddEvenConditional, typename Engine = std::mt19937>
struct two_level_sampler {
  using PathType = typename Action::PathType;

  two_level_sampler(std::shared_ptr<Action> action_,
                    std::shared_ptr<OddEvenConditional> odd_even_conditional_,
                    std::shared_ptr<Engine> engine_)
      : action{std::move(action_)},
        odd_even_conditional{std::move(odd_even_conditional_)}, engine{
                                                                    std::move(engine_)} {}

  std::optional<PathType> perform_step(const PathType &coarse_level_proposal,
                                       const PathType &current_fine_sample) {
    const auto [fine_odd, fine_even] = partition_odd_even(current_fine_sample);

    const auto fine_odd_proposal = odd_even_conditional->sample(coarse_level_proposal);
    const auto fine_proposal = combine_odd_even(fine_odd_proposal, coarse_level_proposal);

    const auto fine_action_diff =
        action->evaluate(fine_proposal) - action->evaluate(current_fine_sample);

    const auto conditional_diff =
        odd_even_conditional->log_density(fine_odd, fine_even) -
        odd_even_conditional->log_density(fine_odd_proposal, coarse_level_proposal);

    auto coarse_action = action->make_coarsened_action();
    const auto coarse_action_diff = coarse_action->evaluate(fine_odd) -
                                    coarse_action->evaluate(coarse_level_proposal);

    const auto delta_S = fine_action_diff + conditional_diff + coarse_action_diff;

    if (delta_S < 0)
      return fine_proposal;

    const auto acceptance_prob = std::exp(-delta_S);
    if (unif_dist(*engine) < acceptance_prob)
      return fine_proposal; // accept
    else
      return {}; // reject
  }

private:
  std::shared_ptr<Action> action;
  std::shared_ptr<OddEvenConditional> odd_even_conditional;

  std::shared_ptr<Engine> engine;
  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
