#include "mlmcpi/proposals/random_walk.hh"
#include "mlmcpi/sampling/distributions.hh"
#include "mlmcpi/sampling/single_level_mcmc.hh"

#include <blaze/Blaze.h>

#include <iostream>
#include <numeric>
#include <random>

struct harmonic_oscillator_action {
  using PathType = blaze::DynamicVector<double>;

  double evaluate(const PathType &path) const {
    // First term is computed separately using periodic BCs
    const auto dxdt = (path[0] - path[path.size() - 1]) / delta_t;
    const auto dxdt2 = dxdt * dxdt;
    double res = (dxdt2 + mu2 * path[0] * path[0]);

    for (std::size_t i = 1; i < path.size(); ++i) {
      const auto dxdt = (path[i] - path[i - 1]) / delta_t;
      const auto dxdt2 = dxdt * dxdt;
      res += dxdt2 + mu2 * path[i] * path[i];
    }
    return 0.5 * m0 * delta_t * res;
  }

  double analytic_solution(std::size_t path_length) {
    double R = 1. + 0.5 * delta_t * delta_t * mu2 -
               delta_t * std::sqrt(mu2) *
                   std::sqrt(1. + 0.25 * delta_t * delta_t * mu2);
    return 1. /
           (2. * m0 * std::sqrt(mu2) *
            std::sqrt(1 + 0.25 * delta_t * delta_t * mu2)) *
           (1. + std::pow(R, path_length)) / (1. - std::pow(R, path_length));
  }

  double delta_t;

  constexpr static double m0 = 1;
  constexpr static double mu2 = 1;
};

int main() {
  std::random_device rd;
  auto engine = std::make_shared<std::mt19937>(rd());

  constexpr double T = 4;
  constexpr double delta_t = 0.5;
  constexpr int N = T / delta_t;
  std::cout << "a = " << delta_t << ", N = " << N << std::endl;

  auto action = std::make_shared<harmonic_oscillator_action>();
  action->delta_t = delta_t;

  blaze::DynamicMatrix<double> Sigma = 0.1 * blaze::IdentityMatrix<double>(N);
  auto proposal_dist =
      std::make_shared<mlmcpi::random_walk_proposal<>>(Sigma, engine);

  mlmcpi::single_level_mcmc sampler(action, proposal_dist);
  auto initial_path = blaze::zero<double>(N);

  constexpr int n_burnin = 1000;
  constexpr int n_samples = 20000;
  auto samples = sampler.sample(n_burnin, n_samples, initial_path);

  auto qoi = [](const blaze::DynamicVector<double> &path) {
    return blaze::mean(path * path);
  };

  auto res = std::transform_reduce(samples.begin(), samples.end(), 0.,
                                   std::plus<double>{}, qoi);
  res /= samples.size();

  std::cout << "Result   = " << res << std::endl;
  std::cout << "Analytic = " << action->analytic_solution(N) << std::endl;
}
