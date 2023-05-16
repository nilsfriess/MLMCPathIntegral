#pragma once

#include "mlmcpi/common/partition.hh"

#include <memory>
#include <optional>
#include <random>

namespace mlmcpi {
template <typename Action, typename CoarseSampler, typename OddEvenConditional,
          typename Engine>
struct two_level_sampler {
  using PathType = typename Action::PathType;

  two_level_sampler(Action &action_, CoarseSampler &coarse_sampler_,
                    OddEvenConditional &odd_even_conditional_, Engine &engine_)
      : action{action_},
        coarse_sampler{coarse_sampler_},
        odd_even_conditional{odd_even_conditional_},
        engine{engine_} {}

  std::optional<PathType> perform_step(const PathType &current) {
    /* Step 1: Generate coarse-level proposal */
    auto [_, current_even] = partition_odd_even(current);
    auto coarse_proposal_opt = coarse_sampler.perform_step(current_even);

    // If coarse proposal is already rejected, we don't even check if it would be accepted
    // but just reject here
    if (not coarse_proposal_opt)
      return {};

    auto coarse_proposal = coarse_proposal_opt.value();

    /* Step 2: "Inform" fine level about the (accepted) coarse-level proposal and perform
     * Metropolis-Hastings step. */
    auto fine_proposal =
        combine_odd_even(odd_even_conditional.sample(coarse_proposal), coarse_proposal);

    const auto fine_action_diff =
        action.evaluate(fine_proposal) - action.evaluate(current);

    const auto conditional_diff = odd_even_conditional.log_density(current) -
                                  odd_even_conditional.log_density(fine_proposal);

    auto coarse_action = action.make_coarsened_action();
    const auto coarse_action_diff =
        coarse_action.evaluate(current_even) - coarse_action.evaluate(coarse_proposal);

    const auto delta_S = fine_action_diff + conditional_diff + coarse_action_diff;

    if (delta_S < 0)
      return fine_proposal;

    const auto acceptance_prob = std::exp(-delta_S);
    if (unif_dist(engine) < acceptance_prob)
      return fine_proposal; // accept
    else
      return {}; // reject
  }

private:
  Action &action;
  CoarseSampler &coarse_sampler;
  OddEvenConditional &odd_even_conditional;

  Engine &engine;
  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
