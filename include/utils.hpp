#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <stdexcept>
#include <type_traits>
#include "concepts.hpp"

template <Numeric T>
class RandomGenerator {
 public:
  static constexpr T defaultMin() { return T{0}; }

  static constexpr T defaultMax() {
    if constexpr (std::integral<T>) {
      return T{100};
    } else {
      return T{1};
    }
  }

  RandomGenerator(T minVal = defaultMin(), T maxVal = defaultMax())
      : minVal(minVal), maxVal(maxVal), rd(), gen(rd()) {
    if (minVal > maxVal) {
      throw std::invalid_argument("minVal > maxVal");
    }

    if constexpr (std::integral<T>) {
      dist = std::uniform_int_distribution<T>(minVal, maxVal);
    } else {
      dist = std::uniform_real_distribution<T>(minVal, maxVal);
    }
  }

  T operator()() { return dist(gen); }

 private:
  T minVal, maxVal;
  std::random_device rd;
  std::mt19937 gen;

  std::conditional_t<std::integral<T>,
                     std::uniform_int_distribution<T>,
                     std::uniform_real_distribution<T>>
      dist;
};

template <Numeric T>
class CudaMemory {
 public:
  explicit CudaMemory(size_t len) : len_(len), size_(sizeof(T) * len) {
    cudaMalloc(&mem_, size_);
  }

  ~CudaMemory() { cudaFree(mem_); }

  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  T* ptr() const { return mem_; }
  size_t getSize() const { return size_; }

 private:
  T* mem_ = nullptr;
  size_t len_ = 0;
  size_t size_ = 0;
};

#endif /* UTILS_H */
