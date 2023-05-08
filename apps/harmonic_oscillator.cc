#include "mlmcpi/proposals/random_walk.hh"
#include "mlmcpi/sampling/distributions.hh"
#include "mlmcpi/sampling/single_level_mcmc.hh"

#include <blaze/Blaze.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <fstream>
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

  double analytic_solution(std::size_t path_length) const {
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

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Provide parameter file as argument" << std::endl;
    return -1;
  }

  std::random_device rd;
  auto engine = std::make_shared<std::mt19937>(rd());

  std::ifstream params_file(argv[1]);
  json data = json::parse(params_file);

  const double T = data["T"];
  const double delta_t = data["delta_t"];
  const int N = T / delta_t;

  auto action = std::make_shared<harmonic_oscillator_action>();
  action->delta_t = delta_t;

  const blaze::DynamicMatrix<double> Sigma =
      0.08 * blaze::IdentityMatrix<double>(N);
  auto proposal =
      std::make_shared<mlmcpi::random_walk_proposal<>>(Sigma, engine);

  mlmcpi::single_level_mcmc sampler(action, proposal);
  auto initial_path = blaze::zero<double>(N);

  const int n_burnin = data["n_burnin"];
  const int n_samples = data["n_samples"];
  auto sample_res = sampler.sample(n_burnin, n_samples, initial_path);
  auto samples = sample_res.samples;
  auto acceptance_rate = sample_res.acceptance_rate;

  auto qoi = [&](const blaze::DynamicVector<double> &path) {
    return blaze::mean(path * path);
  };

  auto mean = (1. / samples.size()) *
              std::transform_reduce(samples.begin(), samples.end(), 0.,
                                    std::plus<double>{}, qoi);

  auto var = (1. / (samples.size() - 1)) *
             std::transform_reduce(
                 samples.begin(), samples.end(), 0., std::plus<double>{},
                 [&](const blaze::DynamicVector<double> &path) {
                   const auto diff = qoi(path) - mean;
                   return diff * diff;
                 });

  std::cout << "Result   = " << mean << " ± " << var << std::endl;
  std::cout << "Analytic = " << action->analytic_solution(N) << std::endl;
  std::cout << "Acceptance rate = " << acceptance_rate << std::endl;
}
