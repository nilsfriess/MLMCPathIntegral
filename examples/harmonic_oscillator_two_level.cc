#include "analytic_solution.hh"
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

#if USE_BLAZE
using Path     = blaze::DynamicVector<double>;
using ZeroPath = blaze::ZeroVector<double>;
#endif

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Provide parameter file as argument" << std::endl;
    return -1;
  }

  std::random_device rd;
  using Engine = std::mt19937_64;
  Engine engine{rd()};

  std::ifstream params_file(argv[1]);
  json params = json::parse(params_file);

  double T      = params["T"];
  std::size_t N = params["N"];

  const std::size_t n_burnin = params["n_burnin"];

  double delta_t = T / N;

  using Action        = harmonic_oscillator_action<Path>;
  using CoarseSampler = hmc_sampler<Action, Engine>;
  using OddEvenCond   = gaussian_even_odd_conditional<Action, Engine>;
  using Sampler       = two_level_sampler<Action, CoarseSampler, OddEvenCond, Engine>;

  Action action{N, delta_t, params["m0"], params["mu2"]};
  auto coarse_action = action.make_coarsened_action();

  CoarseSampler coarse_sampler{0.1, coarse_action, engine};
  OddEvenCond even_odd_conditional{action, engine};

  Sampler sampler{action, coarse_sampler, even_odd_conditional, engine};

  single_level_mcmc mcmc(sampler);

  Path initial_tune_path = ZeroPath(N / 2);
  auto tuned_value =
      coarse_sampler.autotune_stepsize(initial_tune_path, params["hmc_acc_rate"]);
  if (tuned_value)
    std::cout << "Tuned hmc sampler with step size " << tuned_value.value() << "\n";
  else
    std::cout << "Failed to tune hmc sampler\n";

  Path initial_path = ZeroPath(N);
  using QOI         = mean_displacement<Path>;
  const auto result =
      mcmc.run<QOI>(params["n_burnin"], initial_path, params["stat_error"]);

  const auto analytical = analytic_solution(delta_t, params["m0"], params["mu2"], N);

  std::cout << "Result          = " << result.mean() << " ± " << result.mean_error()
            << "\n";
  std::cout << "|Q - Q_{exact}| = " << std::abs(result.mean() - analytical) << "\n";
  std::cout << "Samples         = " << result.num_samples() << "\n";
  std::cout << "Acceptance rate = " << result.acceptance_rate() << "\n";
  std::cout << "Autocorr. time  = " << result.integrated_autocorr_time() << "\n";
}
