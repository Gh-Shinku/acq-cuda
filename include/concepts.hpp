#ifndef CONCEPTS_H
#define CONCEPTS_H

#include <concepts>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

#endif /* CONCEPTS_H */