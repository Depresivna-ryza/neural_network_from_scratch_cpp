#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <tuple>

#include "miscellaneous.hpp"
#include "neuralnetwork.hpp"
#include "tensor.hpp"
#include "matrix.hpp"
#include "function.hpp"

TEST_CASE("Tensor Basics") {
    Tensor t({2, 3});
    REQUIRE(t.get_shape() == std::vector<size_t>({2, 3}));

    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    t.set_data(data);
    REQUIRE(t.get_data() == data);
    REQUIRE(t.get({0, 0}) == 1);
    REQUIRE(t.get({0, 2}) == 3);
    REQUIRE(t.get({1, 1}) == 5);
    REQUIRE(t.get({1, 2}) == 6);

    t.transpose();
    REQUIRE(t.get_shape() == std::vector<size_t>({3, 2}));
    REQUIRE(t.get({0, 0}) == 1);
    REQUIRE(t.get({1, 1}) == 5);
    REQUIRE(t.get({2, 1}) == 6);
    REQUIRE(t.get({2, 0}) == 3);
}


TEST_CASE("Matrix Basics") {
    Matrix m = Matrix::zero_create(2, 3);
    REQUIRE(m.get_data() == std::vector<double>({0, 0, 0, 0, 0, 0}));
    m.set_data(std::vector<double>({1, 2, 3, 4, 5, 6}));
    REQUIRE(m.get_data() == std::vector<double>({1, 2, 3, 4, 5, 6}));
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            REQUIRE(m.get(i, j) == i * m.cols + j + 1);
        }
    }
}

TEST_CASE("Matrix addition") {
    Matrix m1 = Matrix::zero_create(2, 3);
    Matrix m2 = Matrix::zero_create(2, 3);
    m1.set_data(std::vector<double>({1, 2, 3, 4, 5, 6}));
    m2.set_data(std::vector<double>({7, 8, 9, 10, 11, 12}));

    Matrix m3 = m1 + m2;
    REQUIRE(m3.get_data() == std::vector<double>({8, 10, 12, 14, 16, 18}));
    for (size_t i = 0; i < m3.rows; ++i) {
        for (size_t j = 0; j < m3.cols; ++j) {
            REQUIRE(m3.get(i, j) == m1.get(i, j) + m2.get(i, j));
        }
    }

    m3 += 100;
    REQUIRE(m3.get_data() == std::vector<double>({108, 110, 112, 114, 116, 118}));
    for (size_t i = 0; i < m3.rows; ++i) {
        for (size_t j = 0; j < m3.cols; ++j) {
            REQUIRE(m3.get(i, j) == m1.get(i, j) + m2.get(i, j) + 100);
        }
    }

    m3 -= m1;
    m3 -= m2;
    REQUIRE(m3.get_data() == std::vector<double>({100, 100, 100, 100, 100, 100}));
    m3 += m1+m2;

    REQUIRE(m3 - 100 == m1 + m2);
    REQUIRE(m3 - 100 - m1 - m2 == Matrix::zero_create(2, 3));
    REQUIRE(m3 - m1 == m2 + 100);
}

TEST_CASE("Matrix Multiplication") {
    Matrix m1 = Matrix::zero_create(2, 3);
    Matrix m2 = Matrix::zero_create(3, 2);
    m1.set_data(std::vector<double>({1, 2, 3, 4, 5, 6}));
    m2.set_data(std::vector<double>({7, 8, 9, 10, 11, 12}));

    Matrix m3 = m1 * m2;
    REQUIRE(m3.get_shape() == std::tuple<size_t, size_t>({2, 2}));
    REQUIRE(m3.get_data() == std::vector<double>({58, 64, 139, 154}));

    m3 = m2 * m1;
    REQUIRE(m3.get_shape() == std::tuple<size_t, size_t>({3, 3}));
    REQUIRE(m3.get_data() == std::vector<double>({39, 54, 69, 49, 68, 87, 59, 82, 105}));

    m3 = m1 * 2;
    REQUIRE(m3.get_data() == std::vector<double>({2, 4, 6, 8, 10, 12}));
}


TEST_CASE("Matrix layer potential evaluation") {
    Matrix x = Matrix::zero_create(3, 1);
    x.set_data(std::vector<double>({1, 2, 3}));

    Matrix w = Matrix::zero_create(2, 3);
    w.set_data(std::vector<double>({1, 2, 3, 4, 5, 6}));

    Matrix b = Matrix::zero_create(2, 1);
    b.set_data(std::vector<double>({1, 2}));

    Matrix y = w * x + b;

    REQUIRE(y.get_data() == std::vector<double>({15, 34}));
}

TEST_CASE("Matrix ReLU activation function") {
    Matrix x = Matrix::zero_create(10, 1);
    x.set_data(std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    x -= 5;
    x.map(ReLU::evaluate);
    REQUIRE(x.get_data() == std::vector<double>({0, 0, 0, 0, 0, 1, 2, 3, 4, 5}));
}

TEST_CASE("Matrix Sigmoid activation function") {
    Matrix x = Matrix::zero_create(3, 1);
    x.map(Sigmoid::evaluate);
    REQUIRE(x.get_data() == std::vector<double>({0.5, 0.5, 0.5}));
}