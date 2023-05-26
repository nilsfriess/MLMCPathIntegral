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

#if USE_BLAZE
using Path     = blaze::DynamicVector<double>;
using ZeroPath = blaze::ZeroVector<double>;
#endif

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Provide parameter file as argument" << std::endl;
    return -1;
  }

  using Engine = std::mt19937_64;

  std::random_device rd;
  Engine engine{rd()};

  std::ifstream params_file(argv[1]);
  json params = json::parse(params_file);

  const double m0  = params["m0"];
  const double mu2 = params["mu2"];

  const double T       = params["T"];
  const std::size_t N  = params["N"];
  const double delta_t = T / N;

  const auto initial_path = ZeroPath(N);

  using Action = harmonic_oscillator_action<Path>;
  Action action{delta_t, m0, mu2};

  hmc_sampler<Action, Engine> single_step_sampler{0.1, action, engine};
  single_step_sampler.autotune_stepsize(initial_path, 0.8);

  const std::size_t n_burnin  = params["n_burnin"];
  const std::size_t n_samples = params["n_samples"];

  using QOI = mean_displacement<Action::PathType>;
  single_level_mcmc sampler{single_step_sampler};
  const auto result = sampler.run<QOI>(n_burnin, n_samples, initial_path);

  const auto mean     = result.mean();
  const auto mean_err = result.mean_error();
  const auto acc_rate = result.acceptance_rate();
  const auto autocorr = result.integrated_autocorr_time();

  std::cout << "Result   = " << mean << " ± " << mean_err << std::endl;
  std::cout << "Analytic = " << action.analytic_solution(N) << std::endl;
  std::cout << "Acceptance rate = " << acc_rate << std::endl;
  std::cout << "Autocorr. time  = " << autocorr << std::endl;
}
