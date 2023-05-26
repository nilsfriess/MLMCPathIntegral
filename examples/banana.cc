#include <blaze/Blaze.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using Mat = blaze::StaticMatrix<double, 2, 2>;
using Vec = blaze::StaticVector<double, 2>;

template <typename MatrixType, typename VectorType>
[[nodiscard]] inline double normal_density(const VectorType &x, const VectorType &mu,
                                           const MatrixType &sigma) {
  const std::size_t k = x.size();

  auto norm_factor = 1. / std::sqrt(std::pow(2 * M_PI, k) * blaze::det(sigma));
  auto exp_term    = -0.5 * (blaze::trans(x - mu) * blaze::inv(sigma) * (x - mu));

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

template <typename DataT, typename TargetDensity>
std::vector<DataT> random_walk_mcmc(std::size_t n_burnin, std::size_t n_samples,
                                    DataT initial, const TargetDensity &target_density) {
  std::vector<DataT> samples(n_burnin + n_samples);
  samples[0] = initial;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif_dist;

  const Mat Sigma{{1., 0.5}, {0.5, 1}};

  for (std::size_t i = 1; i < n_burnin + n_samples; ++i) {
    const auto current  = samples.at(i - 1);
    const auto proposal = normal_sample(current, Sigma);

    const auto acc_ratio = target_density(proposal) / target_density(current);

    if (unif_dist(generator) <= acc_ratio)
      samples.at(i) = proposal;
    else
      samples.at(i) = current;
  }

  samples.erase(samples.begin(), samples.begin() + static_cast<long>(n_burnin));
  return samples;
}

// See: https://transportmaps.mit.edu/docs/example-banana-2d.html
struct banana_distribution {
  inline double operator()(const Vec &x) const {
    const Mat sigma{{1., 0.9}, {0.9, 1.}};
    const Vec mu{0, 0};

    return normal_density(apply_B_inv(x), mu, sigma);
  }

private:
  Vec apply_B_inv(const Vec &x) const {
    constexpr double a = 1.;
    constexpr double b = 1.;

    const auto x1 = x.at(0);
    const auto x2 = x.at(1);
    return {{x1 / a, a * (x2 + b * (x1 * x1 + a * a))}};
  }
};

int main(int argc, char *argv[]) {
  if (argc != 3) {
    return -1;
  }

  std::size_t n_burnin  = std::atoi(argv[1]);
  std::size_t n_samples = std::atoi(argv[2]);

  banana_distribution target;
  auto samples = random_walk_mcmc(n_burnin, n_samples, Vec{-4, 5}, target);

  for (const auto &sample : samples)
    std::cout << sample[0] << " " << sample[1] << "\n";
}
