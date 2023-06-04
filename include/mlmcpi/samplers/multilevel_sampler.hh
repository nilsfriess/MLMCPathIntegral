#pragma once

#include "mlmcpi/common/partition.hh"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

namespace mlmcpi {
template <typename Action, typename CoarseSampler, typename OddEvenCondFactory,
          typename Engine>
struct multilevel_sampler {
  using PathType = typename Action::PathType;

  multilevel_sampler(std::size_t levels_, Action &coarsest_action_,
                     CoarseSampler &coarse_sampler_,
                     OddEvenCondFactory &odd_even_factory_, Engine &engine_)
      : levels{levels_},
        coarse_sampler{coarse_sampler_},
        engine{engine_} {
    assert(levels > 2);

    actions.push_back(coarsest_action_);
    for (std::size_t l = 1; l < levels; ++l)
      actions.emplace_back(actions.at(l - 1).make_finer_action());

    for (std::size_t l = 1; l < levels; ++l)
      odd_even_conditionals.emplace_back(odd_even_factory_(actions[l]));
  }

  /*
    One step consists of the following:
    - For each level repeat:
      - Sample new coarse modes, given coarse modes (i.e., sample on previous level)
      - If this sample is rejected, stop and reject
      - Otherwise, fill in the fine modes
      - Compute the acceptance probability and perform MH-AR step
   */

  std::optional<PathType> perform_step(const PathType &current) {
    assert(current.size() == actions[levels - 1].get_path_length());

    std::vector<PathType> current_on_level;
    current_on_level.push_back(current);
    for (std::size_t level = 0; level < levels - 1; ++level) {
      const auto [_, coarse_modes] = partition_odd_even(current_on_level[level]);
      current_on_level.push_back(coarse_modes);
    }
    std::reverse(current_on_level.begin(), current_on_level.end());

    assert(current_on_level[0].size() == actions[0].get_path_length());

    // Compute coarse proposal on level 0
    const auto current_proposal_opt = coarse_sampler.perform_step(current_on_level[0]);
    if (not current_proposal_opt)
      return {};

    auto current_proposal = current_proposal_opt.value();

    for (std::size_t prev_level = 0; prev_level < levels - 1; ++prev_level) {
      auto fine_modes = odd_even_conditionals.at(prev_level).sample(current_proposal);
      current_proposal = combine_odd_even(fine_modes, current_proposal);

      if (should_reject(prev_level, current_proposal, current_on_level.at(prev_level + 1),
                        fine_modes))
        return {};
    }

    assert(current.size() == current_proposal.size());
    return current_proposal;
  }

  std::size_t get_finest_path_length() const {
    return actions[levels - 1].get_path_length();
  }

  Action get_action(std::size_t level) const { return actions[level]; }

private:
  bool should_reject(std::size_t level, const PathType &current_proposal,
                     const PathType &current_sample, const PathType &coarse_proposal) {
    auto [_, current_even] = partition_odd_even(current_sample);

    const auto fine_action_diff = actions[level + 1].evaluate(current_proposal) -
                                  actions[level + 1].evaluate(current_sample);

    const auto conditional_diff =
        odd_even_conditionals[level].log_density(current_sample) -
        odd_even_conditionals[level].log_density(current_proposal);

    const auto coarse_action_diff =
        actions[level].evaluate(current_even) - actions[level].evaluate(coarse_proposal);

    const auto delta_S = fine_action_diff + conditional_diff + coarse_action_diff;

    if (delta_S < 0)
      return false; // accept

    const auto acceptance_prob = std::exp(-delta_S);
    if (unif_dist(engine) < acceptance_prob)
      return false; // accept
    else
      return true; // reject
  }

  const std::size_t levels;

  CoarseSampler &coarse_sampler;

  std::vector<Action> actions;
  std::vector<std::invoke_result_t<OddEvenCondFactory, Action>> odd_even_conditionals;

  Engine &engine;
  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
