#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "mlmcpi/monte_carlo/single_level_mcmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include "mlmcpi/samplers/hmc.hh"
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
  const std::size_t N  = params["N"];
  const double delta_t = T / N;

  // std::cout << "Using path length N = " << N << std::endl;
  const auto initial_path = blaze::ZeroVector<double>(N);

  auto action = std::make_shared<harmonic_oscillator_action>(delta_t);
  std::shared_ptr<sampler<harmonic_oscillator_action>> single_step_sampler;

  if (params["sampler"] == "random_walk") {
    const blaze::DynamicMatrix<double> Sigma = 0.08 * blaze::IdentityMatrix<double>(N);
    single_step_sampler =
        std::make_shared<random_walk_sampler<harmonic_oscillator_action>>(Sigma, action,
                                                                          engine);
  } else if (params["sampler"] == "hmc") {
    single_step_sampler =
        std::make_shared<hmc_sampler<harmonic_oscillator_action>>(0.1, action, engine);
    auto tuned_value =
        static_cast<hmc_sampler<harmonic_oscillator_action> *>(single_step_sampler.get())
            ->autotune_stepsize(initial_path, 0.8);
    if (tuned_value)
      std::cout << "Autotuned HMC sampler successfully with dt = " << tuned_value.value()
                << "\n";
    else
      std::cout << "Failed to autotune HMC sampler\n";
  } else {
    std::cerr << "Sampler " << params["sampler"] << " not implemented." << std::endl;
    return -1;
  }

  const std::size_t n_burnin  = params["n_burnin"];
  const std::size_t n_samples = params["n_samples"];

  using QOI = mean_displacement<harmonic_oscillator_action::PathType>;
  single_level_mcmc sampler(single_step_sampler);
  const auto result = sampler.run<QOI>(n_burnin, n_samples, initial_path);

  const auto mean     = result.mean();
  const auto var      = result.variance();
  const auto acc_rate = result.acceptance_rate();
  const auto autocorr = result.integrated_autocorr_time();

  std::cout << "Result   = " << mean << " ± " << var << std::endl;
  std::cout << "Analytic = " << action->analytic_solution(N) << std::endl;
  std::cout << "Acceptance rate = " << acc_rate << std::endl;
  std::cout << "Autocorrelation = " << autocorr << std::endl;
}
