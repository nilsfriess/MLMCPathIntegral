#pragma once

#include "mlmcpi/common/mcmc_result.hh"
#include "mlmcpi/common/partition.hh"
#include "mlmcpi/samplers/two_level_sampler.hh"

#include <memory>
#include <random>

namespace mlmcpi {

template <typename CoarseLevelSampler, typename Action, typename EvenOddConditional,
          typename Engine = std::mt19937>
class two_level_mcmc {
public:
  using PathType = typename Action::PathType;

  two_level_mcmc(std::shared_ptr<CoarseLevelSampler> coarse_sampler_,
                 std::shared_ptr<Action> action_,
                 std::shared_ptr<EvenOddConditional> even_odd_conditional_,
                 std::shared_ptr<Engine> engine_)
      : coarse_sampler{std::move(coarse_sampler_)}, action{std::move(action_)},
        even_odd_conditional{std::move(even_odd_conditional_)},
        engine{std::move(engine_)}, tl_sampler{action, even_odd_conditional, engine} {}

  template <typename QOI>
  mcmc_result<typename QOI::ResultType> run(std::size_t n_burnin, std::size_t n_samples,
                                            PathType initial_path) {
    mcmc_result<typename QOI::ResultType> res(n_samples);
    auto current = initial_path;

    QOI qoi;

    for (std::size_t i = 0; i < n_burnin; ++i) {
      current = single_step(current);
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
      current = single_step(current);
      res.add_sample(qoi(current), true);
    }

    return res;
  }

private:
  PathType single_step(const PathType &current) {
    auto [current_even, current_odd] = partition_odd_even(current);

    const auto coarse_proposal = coarse_sampler->perform_step(current_even);
    if (!coarse_proposal) {
      // Coarse level proposal rejected, return current sample
      return current;
    }

    auto fine_proposal = tl_sampler.perform_step(coarse_proposal.value(), current);
    return fine_proposal.value_or(current);
  }

  std::shared_ptr<CoarseLevelSampler> coarse_sampler;

  std::shared_ptr<Action> action;
  std::shared_ptr<EvenOddConditional> even_odd_conditional;

  std::shared_ptr<Engine> engine;

  two_level_sampler<Action, EvenOddConditional> tl_sampler;
};

} // namespace mlmcpi
