#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace mlmcpi {
template <typename DataT = double> class mcmc_result {
public:
  mcmc_result(std::size_t n_samples) { samples.reserve(n_samples); }

  void add_sample(const DataT &sample, bool was_accepted) {
    samples.push_back(sample);

    if (was_accepted)
      accepted_samples++;
  }

  DataT mean() const {
    const auto sum =
        std::reduce(samples.begin(), samples.end(), DataT{0}, std::plus<DataT>{});
    return 1. / samples.size() * sum;
  }

  DataT variance() const {
    const auto m = mean();
    std::vector<DataT> diff(samples.size());
    std::transform(samples.begin(), samples.end(), diff.begin(),
                   [m](DataT x) { return x - m; });
    const auto sum_sq =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), DataT{0});
    return 1. / (samples.size() - 1) * sum_sq;
  }

  double acceptance_rate() const { return (1. * accepted_samples) / samples.size(); }

  std::size_t integrated_autocorr_time(std::size_t window_size = 200) const {
    const auto m = mean();

    const auto rho = [&](std::size_t s) -> double {
      double sum = 0;
      for (std::size_t j = 1; j < samples.size() - s; ++j)
        sum += (samples[j] - m) * (samples[j + s] - m);
      return 1. / (samples.size() - s) * sum;
    };

    double sum = 0;
    const auto rho_zero = rho(0);
    for (std::size_t s = 1; s < window_size; ++s)
      sum += rho(s) / rho_zero;
    return static_cast<std::size_t>(std::ceil(1 + 2 * sum));
  }

  std::vector<DataT> samples;

private:
  std::size_t accepted_samples = 0;
};
} // namespace mlmcpi
