#ifndef ACQ_CUDA_DEVICE_BUFFER_HPP
#define ACQ_CUDA_DEVICE_BUFFER_HPP

#include "cuda/check.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace acq::cuda {

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) {
      return;
    }
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("DeviceBuffer allocation size overflow");
    }

    ACQ_CUDA_CHECK(cudaMalloc(&data_, bytes()));
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace acq::cuda

#endif  // ACQ_CUDA_DEVICE_BUFFER_HPP
