#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "analytic_solution.hh"
#include "mlmcpi/monte_carlo/single_level_mcmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include "mlmcpi/samplers/hmc.hh"
#include "mlmcpi/samplers/random_walk_sampler.hh"
#include "mlmcpi/samplers/sampler.hh"

#include <blaze/Blaze.h>
#include <cmath>
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

  const double T       = params["T"];
  const std::size_t N  = params["N"];
  const double delta_t = T / N;

  std::cout << "Using stepsize " << delta_t << "\n";

  using Action = harmonic_oscillator_action<Path>;
  Action action{N, delta_t, params["m0"], params["mu2"]};

  hmc_sampler<Action, Engine> single_step_sampler{delta_t, action, engine};
  const auto initial_path = ZeroPath(N);
  const auto tuned_value =
      single_step_sampler.autotune_stepsize(initial_path, params["hmc_acc_rate"]);

  if (tuned_value)
    std::cout << "Tuned hmc sampler with step size " << tuned_value.value() << "\n";
  else
    std::cout << "Failed to tune hmc sampler\n";

  using QOI = mean_displacement<Action::PathType>;
  single_level_mcmc sampler{single_step_sampler};

  const auto result =
      sampler.run<QOI>(params["n_burnin"], initial_path, params["stat_error"]);

  const auto analytical = analytic_solution(delta_t, params["m0"], params["mu2"], N);

  std::cout << "Result          = " << result.mean() << " ± " << result.mean_error()
            << "\n";
  std::cout << "|Q - Q_{exact}| = " << std::abs(result.mean() - analytical) << "\n";
  std::cout << "Samples         = " << result.num_samples() << "\n";
  std::cout << "Acceptance rate = " << result.acceptance_rate() << "\n";
  std::cout << "Autocorr. time  = " << result.integrated_autocorr_time() << "\n";
}
