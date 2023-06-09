#pragma once

#include "mlmcpi/samplers/sampler.hh"

#ifndef USE_BLAZE
#error random_walk_sampler can current only be used with blaze libraryUSE_BLAZE
#endif
#include <blaze/Blaze.h>

#include <optional>
#include <random>

namespace mlmcpi {
template <typename Action, typename MatrixType = blaze::DynamicMatrix<double>,
          typename Engine = std::mt19937>
struct random_walk_sampler : sampler<Action> {
  using PathType = typename Action::PathType;

  random_walk_sampler(const MatrixType &sigma_, Action &action_, Engine &engine_)
      : sigma{sigma_},
        inv_sigma{blaze::inv(sigma)},
        det_sigma{blaze::det(sigma)},
        engine{std::move(engine_)},
        action{std::move(action_)} {
    cholL = sigma;
    blaze::potrf(cholL, 'L'); // Compute Cholesky decomposition of Sigma
  }

  std::optional<PathType> perform_step(const PathType &current) override {
    const auto proposal = generate_proposal(current);

    const auto log_acceptance_prob =
        action.evaluate(proposal) - action.evaluate(current) +
        std::log(density(proposal, current)) - std::log(density(current, proposal));

    if (log_acceptance_prob < 0)
      return proposal;

    auto acceptance_prob = std::exp(-log_acceptance_prob);
    if (unif_dist(engine) < acceptance_prob)
      return proposal;
    else
      return {};
  }

private:
  [[nodiscard]] inline double density(const PathType &eval_point,
                                      const PathType &mean) const {
    const std::size_t k = mean.size();

    auto norm_factor = 1. / std::sqrt(std::pow(2 * M_PI, k) * det_sigma);
    auto exp_term =
        -0.5 * (blaze::trans(eval_point - mean) * inv_sigma * (eval_point - mean));

    return norm_factor * std::exp(exp_term);
  }

  [[nodiscard]] inline PathType generate_proposal(const PathType &mean) {
    blaze::DynamicVector<double> normal_samples(mean.size());
    std::generate(normal_samples.begin(), normal_samples.end(),
                  [&]() { return normal_dist(engine); });
    return mean + blaze::decllow(sigma) * normal_samples;
  }

  MatrixType sigma;
  MatrixType inv_sigma;
  double det_sigma;
  MatrixType cholL; // Lower part of cholesky decomposition of sigma

  Engine engine;
  std::normal_distribution<double> normal_dist;

  Action action;

  std::uniform_real_distribution<double> unif_dist;
};

} // namespace mlmcpi
