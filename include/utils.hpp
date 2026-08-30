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

#endif /* UTILS_H */
