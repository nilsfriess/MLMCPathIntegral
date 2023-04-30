#pragma once

#include <blaze/Blaze.h>
#include <blaze/math/dense/DynamicMatrix.h>
#include <random>
#include <vector>

namespace mlmcpi {

template <typename MatrixType, typename VectorType>
[[nodiscard]] inline double normal_density(const VectorType &x,
                                           const VectorType &mu,
                                           const MatrixType &sigma) {
  const std::size_t k = x.size();

  auto norm_factor = 1. / std::sqrt(std::pow(2 * M_PI, k) * blaze::det(sigma));
  auto exp_term = -0.5 * (blaze::trans(x - mu) * blaze::inv(sigma) * (x - mu));

  return norm_factor * std::exp(exp_term);
}

template <typename MatrixType, typename VectorType>
[[nodiscard]] inline VectorType normal_sample(const VectorType &mu,
                                              const MatrixType &sigma) {
  blaze::DynamicMatrix<double> S = sigma;
  blaze::potrf(S, 'L'); // Compute Cholesky decomposition of Sigma

  std::random_device rd{};
  std::mt19937 engine{rd()};
  std::normal_distribution<> dist;
  auto generator = [&dist, &engine]() { return dist(engine); };

  blaze::DynamicVector<double> normal_samples(mu.size());
  std::generate(normal_samples.begin(), normal_samples.end(), generator);

  return mu + blaze::decllow(S) * normal_samples;
}

} // namespace mlmcpi
