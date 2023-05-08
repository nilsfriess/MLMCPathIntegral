#pragma once

#include <blaze/math/dense/DynamicMatrix.h>
#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/expressions/DMatDetExpr.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/lapack/potrf.h>

#include <memory>
#include <random>

namespace mlmcpi {
template <typename VectorType = blaze::DynamicVector<double>,
          typename MatrixType = blaze::DynamicMatrix<double>,
          typename Engine = std::mt19937>
struct random_walk_proposal {
  random_walk_proposal(const MatrixType &sigma,
                       const std::shared_ptr<Engine> &rand_engine)
      : sigma(sigma), inv_sigma(blaze::inv(sigma)),
        det_sigma(blaze::det(sigma)), engine(std::move(rand_engine)) {
    cholL = sigma;
    blaze::potrf(cholL, 'L'); // Compute Cholesky decomposition of Sigma
  }

  [[nodiscard]] inline double density(const VectorType &eval_point,
                                      const VectorType &mean) const {
    const std::size_t k = mean.size();

    auto norm_factor = 1. / std::sqrt(std::pow(2 * M_PI, k) * det_sigma);
    auto exp_term = -0.5 * (blaze::trans(eval_point - mean) * inv_sigma *
                            (eval_point - mean));

    return norm_factor * std::exp(exp_term);
  }

  [[nodiscard]] inline VectorType sample(const VectorType &mean) {
    blaze::DynamicVector<double> normal_samples(mean.size());
    std::generate(normal_samples.begin(), normal_samples.end(),
                  [&]() { return normal_dist(*engine); });
    return mean + blaze::decllow(sigma) * normal_samples;
  }

private:
  MatrixType sigma;
  MatrixType inv_sigma;
  double det_sigma;
  MatrixType cholL; // Lower part of cholesky decomposition of sigma

  std::shared_ptr<Engine> engine;
  std::normal_distribution<double> normal_dist;
};

} // namespace mlmcpi
