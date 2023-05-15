#include "mlmcpi/actions/harmonic_oscillator.hh"
#include "mlmcpi/distributions/gaussian_even_odd_conditional.hh"
#include "mlmcpi/monte_carlo/two_level_mcmc.hh"
#include "mlmcpi/proposals/hmc.hh"
#include "mlmcpi/qoi/mean_displacement.hh"
#include <blaze/math/sparse/ZeroVector.h>
#include <memory>

using namespace mlmcpi;

int main() {
  std::random_device rd;
  auto engine = std::make_shared<std::mt19937>(rd());

  double T           = 4;
  std::size_t points = 32;

  double delta_t = points / T;

  auto action = std::make_shared<harmonic_oscillator_action>(delta_t);

  auto coarse_sampler =
      std::make_shared<hmc_sampler<harmonic_oscillator_action>>(action, engine);

  auto even_odd_conditional =
      std::make_shared<gaussian_even_odd_conditional<harmonic_oscillator_action>>(action,
                                                                                  engine);

  two_level_mcmc mcmc(coarse_sampler, action, even_odd_conditional, engine);

  const auto initial_path = blaze::ZeroVector<double>(points);

  using QOI = mean_displacement<harmonic_oscillator_action::PathType>;
  auto res  = mcmc.run<QOI>(1000, 10000, initial_path);

  std::cout << "Mean = " << res.mean() << std::endl;
}
