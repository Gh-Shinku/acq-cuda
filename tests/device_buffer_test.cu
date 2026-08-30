#include <catch2/catch_test_macros.hpp>

#include "cuda/device_buffer.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

TEST_CASE("DeviceBuffer owns CUDA allocations", "[cuda][device_buffer]") {
  acq::cuda::DeviceBuffer<float> empty(0);
  CHECK(empty.data() == nullptr);
  CHECK(empty.size() == 0);
  CHECK(empty.bytes() == 0);

  REQUIRE_THROWS_AS(
      acq::cuda::DeviceBuffer<float>(std::numeric_limits<std::size_t>::max()),
      std::overflow_error);

  acq::cuda::DeviceBuffer<float> original(16);
  float* const allocation = original.data();
  REQUIRE(allocation != nullptr);
  CHECK(original.size() == 16);
  CHECK(original.bytes() == 16 * sizeof(float));

  acq::cuda::DeviceBuffer<float> moved(std::move(original));
  CHECK(original.data() == nullptr);
  CHECK(original.size() == 0);
  CHECK(moved.data() == allocation);

  acq::cuda::DeviceBuffer<float> reassigned(1);
  reassigned = std::move(moved);
  CHECK(moved.data() == nullptr);
  CHECK(moved.size() == 0);
  CHECK(reassigned.data() == allocation);
  CHECK(reassigned.size() == 16);
}
