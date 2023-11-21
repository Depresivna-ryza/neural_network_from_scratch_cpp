#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <ostream>
#include <vector>

struct Matrix {
    size_t rows;
    size_t cols;
    std::vector<double> data;

    Matrix(size_t r, size_t c) : rows(r), cols(c) { data.resize(rows * cols); }
    Matrix() = delete;

    void set_data(std::vector<double> d) {
        assert(d.size() == data.size());
        data = d;
    }
    std::vector<double> get_data() const { return data; }

    double get(size_t i, size_t j) const { return data[i * cols + j]; }

    void set(size_t i, size_t j, double value) { data[i * cols + j] = value; }

    void transpose() {
        std::swap(rows, cols);
        std::vector<double> new_data(data.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                new_data[i * cols + j] = data[j * rows + i];
            }
        }
        data = new_data;
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
};
#endif  // MATRIX_H