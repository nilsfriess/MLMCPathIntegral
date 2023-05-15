#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "mlmcpi/monte_carlo/single_level_mcmc.hh"
#include "mlmcpi/proposals/hmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include "mlmcpi/samplers/random_walk_sampler.hh"
#include "mlmcpi/samplers/sampler.hh"

#include <blaze/Blaze.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

using namespace mlmcpi;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Provide parameter file as argument" << std::endl;
    return -1;
  }

  std::random_device rd;
  auto engine = std::make_shared<std::mt19937>(rd());

  std::ifstream params_file(argv[1]);
  json params = json::parse(params_file);

  const double T       = params["T"];
  const double delta_t = params["delta_t"];
  const std::size_t N  = T / delta_t;

  // std::cout << "Using path length N = " << N << std::endl;

  auto action     = std::make_shared<harmonic_oscillator_action>();
  action->delta_t = delta_t;

  std::shared_ptr<sampler<harmonic_oscillator_action>> single_step_sampler;

  if (params["sampler"] == "random_walk") {
    const blaze::DynamicMatrix<double> Sigma = 0.08 * blaze::IdentityMatrix<double>(N);
    single_step_sampler =
        std::make_shared<random_walk_sampler<harmonic_oscillator_action>>(Sigma, action,
                                                                          engine);
  } else if (params["sampler"] == "hmc") {
    single_step_sampler =
        std::make_shared<hmc_sampler<harmonic_oscillator_action>>(action, engine);
  } else {
    std::cerr << "Sampler " << params["sampler"] << " not implemented." << std::endl;
    return -1;
  }

  const std::size_t n_burnin  = params["n_burnin"];
  const std::size_t n_samples = params["n_samples"];
  const auto initial_path     = blaze::ZeroVector<double>(N);

  using QOI = mean_displacement<harmonic_oscillator_action::PathType>;
  single_level_mcmc sampler(single_step_sampler);
  const auto result = sampler.run<QOI>(n_burnin, n_samples, initial_path);

  const auto mean     = result.mean();
  const auto var      = result.variance();
  const auto acc_rate = result.acceptance_rate();

  std::cout << "Result   = " << mean << " ± " << var << std::endl;
  std::cout << "Analytic = " << action->analytic_solution(N) << std::endl;
  std::cout << "Acceptance rate = " << acc_rate << std::endl;
}
