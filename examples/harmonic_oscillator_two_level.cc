#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "mlmcpi/common/partition.hh"
#include "mlmcpi/distributions/gaussian_even_odd_conditional.hh"
#include "mlmcpi/monte_carlo/single_level_mcmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include "mlmcpi/samplers/hmc.hh"
#include "mlmcpi/samplers/two_level_sampler.hh"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/sparse/ZeroVector.h>

#include <fstream>
#include <memory>

using namespace mlmcpi;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Provide parameter file as argument" << std::endl;
    return -1;
  }

  std::random_device rd;
  using Engine = std::mt19937_64;
  auto engine  = std::make_shared<Engine>(rd());

  std::ifstream params_file(argv[1]);
  json params = json::parse(params_file);

  double T      = params["T"];
  std::size_t N = params["N"];

  const std::size_t n_burnin  = params["n_burnin"];
  const std::size_t n_samples = params["n_samples"];

  double delta_t = T / N;

  using Action        = harmonic_oscillator_action;
  using CoarseSampler = hmc_sampler<Action, Engine>;
  using OddEvenCond   = gaussian_even_odd_conditional<Action, Engine>;
  using Sampler       = two_level_sampler<Action, CoarseSampler, OddEvenCond, Engine>;

  auto action = std::make_shared<Action>(delta_t);

  std::shared_ptr<Action> coarse_action = action->make_coarsened_action();

  auto coarse_sampler = std::make_shared<CoarseSampler>(0.1, coarse_action, engine);

  auto even_odd_conditional = std::make_shared<OddEvenCond>(action, engine);

  auto sampler =
      std::make_shared<Sampler>(action, coarse_sampler, even_odd_conditional, engine);

  single_level_mcmc mcmc(sampler);

  blaze::DynamicVector<double> initial_path = blaze::ZeroVector<double>(N);
  auto tuned_value = coarse_sampler->autotune_stepsize(initial_path, 0.8);
  if (tuned_value)
    std::cout << "Autotuned HMC sampler successfully with dt = " << tuned_value.value()
              << "\n";
  else
    std::cout << "Failed to autotune HMC sampler\n";

  using QOI = mean_displacement<harmonic_oscillator_action::PathType>;
  auto res  = mcmc.run<QOI>(n_burnin, n_samples, initial_path);

  std::cout << "Mean      = " << res.mean() << std::endl;
  std::cout << "Var       = " << res.variance() << std::endl;
  std::cout << "acc_rate  = " << res.acceptance_rate() << std::endl;
  std::cout << "auto_corr = " << res.integrated_autocorr_time() << std::endl;
}
