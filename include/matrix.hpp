#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "concepts.hpp"
#include "utils.hpp"

template <Numeric T>
class Matrix {
 private:
  int rows, cols;
  std::vector<T> data;

 public:
  Matrix(int r, int c, T initVal = T{})
      : rows(r), cols(c), data(r * c, initVal) {}

  Matrix(int r, int c, RandomGenerator<T>& rg) : rows(r), cols(c), data(r * c) {
    std::generate(data.begin(), data.end(), [&]() { return rg(); });
  }

  T* ptr() { return data.data(); }
  const T* ptr() const { return data.data(); }

  T& operator()(int i, int j) { return data[i * cols + j]; }
  const T& operator()(int i, int j) const { return data[i * cols + j]; }

  T& at(int i, int j) { return data[i * cols + j]; }
  const T& at(int i, int j) const { return data[i * cols + j]; }

  int row() const { return rows; }
  int col() const { return cols; }

  Matrix operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("matrix dimensions must match");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] + other.data[i];
    }

    return result;
  }

  Matrix operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("matrix dimensions must match");
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] - other.data[i];
    }

    return result;
  }

  Matrix operator*(const Matrix& other) const {
    if (cols != other.rows) {
      throw std::invalid_argument("left matrix columns must match right matrix rows");
    }

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < other.cols; ++j) {
        for (int k = 0; k < cols; ++k) {
          result(i, j) += at(i, k) * other(k, j);
        }
      }
    }

    return result;
  }

  bool operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
      return false;
    }

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (at(i, j) != other(i, j)) {
          return false;
        }
      }
    }
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "[";
    for (int i = 0; i < matrix.rows; ++i) {
      if (i > 0) {
        os << ",\n ";
      }

      os << "[";
      for (int j = 0; j < matrix.cols; ++j) {
        if (j > 0) {
          os << ", ";
        }
        os << matrix(i, j);
      }
      os << "]";
    }
    os << "]";

    return os;
  }
};

#endif /* MATRIX_H */
