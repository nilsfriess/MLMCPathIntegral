#pragma once

#include <algorithm>
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
    const auto sum = std::reduce(samples.begin(), samples.end(), DataT{0},
                                 std::plus<DataT>{});
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

  double acceptance_rate() const {
    return (1. * accepted_samples) / samples.size();
  }

private:
  std::vector<DataT> samples;

  std::size_t accepted_samples = 0;
};
} // namespace mlmcpi
