#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace mlmcpi {
template <typename DataT = double> class mcmc_result {
public:
  mcmc_result() {}

  void add_sample(const DataT &sample, bool was_accepted) {
    total_samples++;

    samples.push_back(sample);

    if (was_accepted)
      accepted_samples++;
  }

  DataT mean(DataT zero = DataT{0}) const {
    const auto sum =
        std::reduce(samples.begin(), samples.end(), zero, std::plus<DataT>{});
    return 1. / total_samples * sum;
  }

  DataT mean_error() const {
    return std::sqrt(((1.0 * integrated_autocorr_time()) / total_samples) * variance());
  }

  DataT variance() const {
    const auto m = mean();
    std::vector<DataT> diff(total_samples);
    std::transform(samples.begin(), samples.end(), diff.begin(),
                   [m](DataT x) { return x - m; });
    const auto sum_sq =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), DataT{0});
    return 1. / (total_samples - 1) * sum_sq;
  }

  double acceptance_rate() const { return (1. * accepted_samples) / total_samples; }

  std::size_t integrated_autocorr_time(std::size_t window_size = 30) const {
    if (window_size > total_samples)
      return total_samples;

    const auto m = mean();

    const auto rho = [&](std::size_t s) -> double {
      double sum = 0;
      for (std::size_t j = 1; j < total_samples - s; ++j)
        sum += (samples[j] - m) * (samples[j + s] - m);
      return 1. / (total_samples - s) * sum;
    };

    double sum = 0;
    const auto rho_zero = rho(0);
    for (std::size_t s = 1; s < window_size; ++s)
      sum += rho(s) / rho_zero;
    const auto tau = static_cast<std::size_t>(std::ceil(1 + 2 * sum));
    return std::max(1UL, tau);
  }

  std::size_t num_samples() const { return total_samples; }

  std::vector<DataT> samples;

private:
  std::size_t total_samples = 0;
  std::size_t accepted_samples = 0;
};
} // namespace mlmcpi
