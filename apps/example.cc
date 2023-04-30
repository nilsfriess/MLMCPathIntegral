#include "mlmcpi/sampling/distributions.hh"
#include "mlmcpi/sampling/mcmc.hh"

#include <blaze/Blaze.h>

#include <cmath>
#include <iostream>
#include <random>

using namespace mlmcpi;

using Mat = blaze::DynamicMatrix<double>;
using Vec = blaze::DynamicVector<double>;

// See: https://transportmaps.mit.edu/docs/example-banana-2d.html
struct banana_distribution {
  inline double evaluate(const Vec &x) {
    const Mat sigma{{1., 0.9}, {0.9, 1.}};
    const Vec mu{0, 0};

    return normal_density(apply_B_inv(x), mu, sigma);
  }

private:
  Vec apply_B_inv(const Vec &x) {
    constexpr double a = 1.;
    constexpr double b = 1.;

    const auto x1 = x.at(0);
    const auto x2 = x.at(1);
    return {{x1 / a, a * (x2 + b * (x1 * x1 + a * a))}};
  }
};

struct proposal_dist {
  inline double evaluate(Vec) const { return 1; }

  inline Vec sample(Vec x) {
    const Mat sigma{{1., 0.5}, {0.5, 1.}};

    return normal_sample(x, sigma);
  }
};

int main() {
  constexpr int n_burnin = 1000;
  constexpr int n_samples = 10000;

  auto sampler = mcmc<Vec, banana_distribution, proposal_dist>{};
  auto samples = sampler.sample(n_burnin, n_samples, {0, 0});

  for (const auto &sample : samples)
    std::cout << sample[0] << " " << sample[1] << "\n";
}
