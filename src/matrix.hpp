#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <ostream>
#include <vector>

#include "miscellaneous.hpp"

struct Matrix {
    size_t rows;  // for weights: number of neurons in output layer
    size_t cols;  // for weights: number of neurons in input layer
    std::vector<double> data;

    Matrix(size_t r, size_t c) : rows(r), cols(c) { data.resize(rows * cols); }
    Matrix(size_t r, size_t c, std::vector<double> d) : rows(r), cols(c), data(d) { assert(d.size() == rows * cols); }
    Matrix(std::vector<double> d) : rows(d.size()), cols(1), data(d) {}
    Matrix() = delete;

    auto get_shape() const { return std::make_tuple(rows, cols); }

    void set_data(std::vector<double> d) {
        assert(d.size() == data.size());
        data = d;
    }
    std::vector<double> get_data() const { return data; }

    double get(size_t row, size_t col) const {
        assert(row < rows && col < cols);
        return data[row * cols + col];
    }
    double& get(size_t row, size_t col) {
        assert(row < rows && col < cols);
        return data[row * cols + col];
    }

    void set(size_t row, size_t col, double value) {
        assert(row < rows && col < cols);
        data[row * cols + col] = value;
    }
    size_t size() const { return data.size(); }

    Matrix& transpose() {
        std::swap(rows, cols);
        std::vector<double> new_data(data.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                new_data[i * cols + j] = data[j * rows + i];
            }
        }
        data = new_data;
        return *this;
    }

    static Matrix zero_create(size_t r, size_t c) {
        Matrix result(r, c);
        result.clear();
        return result;
    }

    // for initialization of a matrix of weights or biases with parameter of the size of last layer
    static Matrix normal_he_create(size_t r, size_t c, double last_layer_size) {
        Matrix result(r, c);
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] = normal_he(last_layer_size);
        }
        return result;
    }
    void clear() {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = 0;
        }
    }

    void map(auto f) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = f(data[i]);
        }
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result = *this;
        result += other;
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    // broadcasted addition
    Matrix operator+(const double& scalar) const {
        Matrix result = *this;
        result += scalar;
        return result;
    }
    Matrix& operator+=(const double& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result = *this;
        result -= other;
        return result;
    }

    Matrix& operator-=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    // broadcasted subtraction
    Matrix operator-(const double& scalar) const {
        Matrix result = *this;
        result -= scalar;
        return result;
    }

    Matrix& operator-=(const double& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= scalar;
        }
        return *this;
    }

    // dot product
    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);
        std::vector<double> new_data(rows * other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum{};
                for (size_t k = 0; k < cols; ++k) {
                    sum += get(i, k) * other.get(k, j);
                }
                new_data[i * other.cols + j] = sum;
            }
        }
        Matrix result(rows, other.cols);
        result.set_data(new_data);
        return result;
    }

    // scalar multiplication
    Matrix operator*(const double& scalar) const {
        Matrix result = *this;
        result *= scalar;
        return result;
    }

    Matrix& operator*=(const double& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "Matrix<" << m.rows << "x" << m.cols << ">{\n";
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                os << m.get(i, j) << " ";
            }
            os << "\n";
        }
        os << "}\n";
        return os;
    }

    friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.rows != rhs.rows) {
            return false;
        }
        if (lhs.cols != rhs.cols) {
            return false;
        }
        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < lhs.cols; ++j) {
                if (lhs.get(i, j) != rhs.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    friend std::partial_ordering operator<=>(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.rows != rhs.rows) {
            return std::partial_ordering::unordered;
        }
        if (lhs.cols != rhs.cols) {
            return std::partial_ordering::unordered;
        }
        
        if (lhs.get(0,0) < rhs.get(0,0)) {
            for (size_t i = 0; i < lhs.rows; ++i) {
                for (size_t j = 0; j < lhs.cols; ++j) {
                    if (lhs.get(i, j) >= rhs.get(i, j)) {
                        return std::partial_ordering::unordered;
                    }
                }
            }
            return std::partial_ordering::less;
        }

        if (lhs.get(0,0) > rhs.get(0,0)) {
            for (size_t i = 0; i < lhs.rows; ++i) {
                for (size_t j = 0; j < lhs.cols; ++j) {
                    if (lhs.get(i, j) <= rhs.get(i, j)) {
                        return std::partial_ordering::unordered;
                    }
                }
            }
            return std::partial_ordering::greater;
        }

        for (size_t i = 0; i < lhs.rows; ++i) {
            for (size_t j = 0; j < lhs.cols; ++j) {
                if (lhs.get(i, j) != rhs.get(i, j)) {
                    return std::partial_ordering::unordered;
                }
            }
        }
        return std::partial_ordering::equivalent;

    }
};
#endif  // MATRIX_H