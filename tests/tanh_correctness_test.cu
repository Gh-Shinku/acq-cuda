#include <catch2/catch_test_macros.hpp>

#include "cuda/check.hpp"
#include "cuda/device_buffer.hpp"
#include "ops/tanh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<float> run_tanh(const std::vector<float>& input) {
  if (input.empty()) {
    return {};
  }

  acq::cuda::DeviceBuffer<float> device_input(input.size());
  acq::cuda::DeviceBuffer<float> device_output(input.size());
  std::size_t const bytes = input.size() * sizeof(float);
  ACQ_CUDA_CHECK(cudaMemcpy(device_input.data(), input.data(), bytes,
                            cudaMemcpyHostToDevice));

  ops::launch_tanh(device_input.data(), device_output.data(), input.size());
  ACQ_CUDA_CHECK(cudaGetLastError());
  ACQ_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(input.size());
  ACQ_CUDA_CHECK(cudaMemcpy(output.data(), device_output.data(), bytes,
                            cudaMemcpyDeviceToHost));
  return output;
}

bool matches_tanh(float expected, float actual) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && std::signbit(expected) == std::signbit(actual);
  }

  constexpr float kAbsoluteTolerance = 1.0e-7f;
  constexpr float kRelativeTolerance = 1.0e-6f;
  float const error = std::abs(expected - actual);
  float const tolerance =
      std::max(kAbsoluteTolerance, kRelativeTolerance * std::abs(expected));
  return error <= tolerance;
}

std::vector<std::pair<std::string, std::vector<float>>> make_cases() {
  std::mt19937 generator(20260830);
  std::uniform_real_distribution<float> uniform(-10.0f, 10.0f);
  std::normal_distribution<float> normal;

  std::vector<std::pair<std::string, std::vector<float>>> cases;
  cases.emplace_back("empty", std::vector<float>{});
  cases.emplace_back("special_values",
                     std::vector<float>{-INFINITY, -100.0f, -20.0f, -1.0f,
                                        -0.0f,    0.0f,   1.0f,   20.0f,
                                        100.0f,   INFINITY, NAN});

  for (std::size_t length : {1U, 255U, 256U, 257U, 1023U, 1024U, 1025U}) {
    std::vector<float> values(length);
    for (float& value : values) {
      value = uniform(generator);
    }
    cases.emplace_back("uniform_boundary_" + std::to_string(length),
                       std::move(values));
  }

  std::vector<float> normal_values(4099);
  for (float& value : normal_values) {
    value = normal(generator);
  }
  cases.emplace_back("normal_4099", std::move(normal_values));
  return cases;
}

}  // namespace

TEST_CASE("CUDA tanh matches the C++ reference", "[tanh][correctness]") {
  for (const auto& [name, input] : make_cases()) {
    std::vector<float> actual = run_tanh(input);
    REQUIRE(actual.size() == input.size());

    for (std::size_t index = 0; index < input.size(); ++index) {
      float const expected = std::tanh(input[index]);
      INFO("case='" << name << "' index=" << index << " input=" << input[index]
                    << " expected=" << expected << " actual=" << actual[index]);
      CHECK(matches_tanh(expected, actual[index]));
    }
  }
}
