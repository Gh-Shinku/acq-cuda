#include <catch2/catch_test_macros.hpp>

#include "ops/tanh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(&ptr_, count * sizeof(float)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* get() const { return ptr_; }

 private:
  float* ptr_ = nullptr;
};

std::vector<float> run_tanh(const std::vector<float>& input) {
  if (input.empty()) {
    return {};
  }

  DeviceBuffer device_input(input.size());
  DeviceBuffer device_output(input.size());
  std::size_t const bytes = input.size() * sizeof(float);
  check_cuda(cudaMemcpy(device_input.get(), input.data(), bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy host-to-device");

  ops::launch_tanh(device_input.get(), device_output.get(), input.size());
  check_cuda(cudaGetLastError(), "launch_tanh");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<float> output(input.size());
  check_cuda(cudaMemcpy(output.data(), device_output.get(), bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy device-to-host");
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
