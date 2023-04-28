#include "mlmcpi/sampling/mcmc.hh"

#include <cmath>
#include <iostream>
#include <random>

using namespace mlmcpi;

struct target_dist {
  inline double evaluate(double x) const {
    return std::exp(-0.5 * ((4 - x * x) * (4 - x * x) + x * x));
  }
};

struct proposal_dist {
  proposal_dist() : generator{std::random_device{}()}, distribution(0, 1.5) {}

  inline double evaluate(double x) const { return x; }

  inline double sample(double x) { return distribution(generator) + x; }

private:
  std::mt19937 generator;
  std::normal_distribution<double> distribution;
};

int main() {
  constexpr int n_burnin = 5000;
  constexpr int n_samples = 10000;

  auto sampler = mcmc<target_dist, proposal_dist>{};
  auto samples = sampler.sample(n_burnin, n_samples);

  for (const auto sample : samples)
    std::cout << sample << "\n";
}
