#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "mlmcpi/common/partition.hh"
#include "mlmcpi/distributions/gaussian_even_odd_conditional.hh"
#include "mlmcpi/monte_carlo/single_level_mcmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include "mlmcpi/samplers/hmc.hh"
#include "mlmcpi/samplers/two_level_sampler.hh"

#include <blaze/math/dense/DynamicVector.h>
#include <blaze/math/sparse/ZeroVector.h>
#include <memory>

using namespace mlmcpi;

int main() {
  std::random_device rd;
  using Engine = std::mt19937_64;
  auto engine  = std::make_shared<Engine>(rd());

  double T           = 4;
  std::size_t points = 64;

  double delta_t = T / points;

  using Action        = harmonic_oscillator_action;
  using CoarseSampler = hmc_sampler<Action, Engine>;
  using OddEvenCond   = gaussian_even_odd_conditional<Action, Engine>;
  using Sampler       = two_level_sampler<Action, CoarseSampler, OddEvenCond, Engine>;

  auto action                           = std::make_shared<Action>(delta_t);
  std::shared_ptr<Action> coarse_action = action->make_coarsened_action();

  auto coarse_sampler = std::make_shared<CoarseSampler>(0.1, coarse_action, engine);

  auto even_odd_conditional = std::make_shared<OddEvenCond>(action, engine);

  auto sampler =
      std::make_shared<Sampler>(action, coarse_sampler, even_odd_conditional, engine);

  single_level_mcmc mcmc(sampler);

  blaze::DynamicVector<double> initial_path = blaze::ZeroVector<double>(points);
  auto tuned_value = coarse_sampler->autotune_stepsize(initial_path);
  if (tuned_value)
    std::cout << "Autotuned HMC sampler successfully with dt = " << tuned_value.value()
              << "\n";
  else
    std::cout << "Failed to autotune HMC sampler\n";

  using QOI = mean_displacement<harmonic_oscillator_action::PathType>;
  auto res  = mcmc.run<QOI>(1000, 10000, initial_path);

  std::cout << "Mean = " << res.mean() << std::endl;
  std::cout << "Var  = " << res.variance() << std::endl;
  std::cout << "AP   = " << res.acceptance_rate() << std::endl;
}
