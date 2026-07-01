#include "matrix.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

TEST_CASE("Matrix Add", "[matrix]") {
  Matrix<int> m_a(4, 5, 1);
  Matrix<int> m_b(4, 5, 2);
  Matrix<int> m_c(4, 5, 3);
  REQUIRE(m_a + m_b == m_c);
}

TEST_CASE("Matrix Minus", "[matrix]") {
  Matrix<int> m_a(4, 5, 1);
  Matrix<int> m_b(4, 5, 2);
  Matrix<int> m_c(4, 5, -1);
  REQUIRE(m_a - m_b == m_c);
}

TEST_CASE("Matrix Multiplication", "[matrix]") {
  Matrix<int> m_a(2, 3);
  Matrix<int> m_b(3, 2);
  Matrix<int> expected(2, 2);

  m_a(0, 0) = 1;
  m_a(0, 1) = 2;
  m_a(0, 2) = 3;
  m_a(1, 0) = 4;
  m_a(1, 1) = 5;
  m_a(1, 2) = 6;

  m_b(0, 0) = 7;
  m_b(0, 1) = 8;
  m_b(1, 0) = 9;
  m_b(1, 1) = 10;
  m_b(2, 0) = 11;
  m_b(2, 1) = 12;

  expected(0, 0) = 58;
  expected(0, 1) = 64;
  expected(1, 0) = 139;
  expected(1, 1) = 154;

  REQUIRE(m_a * m_b == expected);
}

TEST_CASE("Matrix Multiplication with floating point values", "[matrix]") {
  Matrix<double> m_a(2, 3);
  Matrix<double> m_b(3, 2);

  m_a(0, 0) = 1.5;
  m_a(0, 1) = -2.0;
  m_a(0, 2) = 0.25;
  m_a(1, 0) = 3.0;
  m_a(1, 1) = 4.5;
  m_a(1, 2) = -1.25;

  m_b(0, 0) = 2.0;
  m_b(0, 1) = -1.0;
  m_b(1, 0) = 0.5;
  m_b(1, 1) = 3.0;
  m_b(2, 0) = -4.0;
  m_b(2, 1) = 0.25;

  Matrix<double> result = m_a * m_b;

  REQUIRE(result.row() == 2);
  REQUIRE(result.col() == 2);
  REQUIRE(result(0, 0) == Catch::Approx(1.0));
  REQUIRE(result(0, 1) == Catch::Approx(-7.4375));
  REQUIRE(result(1, 0) == Catch::Approx(13.25));
  REQUIRE(result(1, 1) == Catch::Approx(10.1875));
}

TEST_CASE("Matrix formats as NumPy-like rows", "[matrix]") {
  Matrix<int> m(2, 3);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  std::ostringstream oss;
  oss << m;

  REQUIRE(oss.str() == "[[1, 2, 3],\n [4, 5, 6]]");
}

TEST_CASE("Matrix formatting preserves ostream numeric formatting", "[matrix]") {
  Matrix<double> m(2, 2);
  m(0, 0) = 1.5;
  m(0, 1) = -2.0;
  m(1, 0) = 0.25;
  m(1, 1) = 3.75;

  std::ostringstream oss;
  oss << m;

  REQUIRE(oss.str() == "[[1.5, -2],\n [0.25, 3.75]]");
}
