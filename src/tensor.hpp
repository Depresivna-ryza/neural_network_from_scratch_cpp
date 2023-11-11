#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <cassert>
#include <ostream>
#include <vector>

struct Tensor {
    std::vector<size_t> shape;
    std::vector<double> data;

    Tensor(std::vector<size_t> s) : shape(s) {
        size_t size = 1;
        for (size_t i = 0; i < s.size(); ++i) {
            size *= s[i];
        }
        data.resize(size);
    }
    Tensor() = delete;

    std::vector<size_t> get_shape() const { return shape; }

    void set_data(std::vector<double> d) {
        assert(d.size() == data.size());
        data = d;
    }
    std::vector<double> get_data() const { return data; }

    double get(std::vector<size_t> index) const { return data[get_index(index)]; }

    void set(std::vector<size_t> index, double value) { data[get_index(index)] = value; }

    void transpose() {
        assert(shape.size() == 2);
        std::swap(shape[0], shape[1]);
        std::vector<double> new_data(data.size());
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                new_data[i * shape[1] + j] = data[j * shape[0] + i];
            }
        }
        data = new_data;
    }

    Tensor operator+(const Tensor& other) const {
        Tensor result = *this;
        result += other;
        return result;
    }
    Tensor& operator+=(const Tensor& other) {
        assert(shape == other.shape);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    // broadcasted addition
    Tensor operator+(const double& scalar) const {
        Tensor result = *this;
        result += scalar;
        return result;
    }
    Tensor& operator+=(const double& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    // dot product
    double operator*(Tensor const& other) const {
        assert(shape == other.shape);
        double result{};
        for (size_t i = 0; i < data.size(); ++i) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    // scalar multiplication
    Tensor operator*(const double& scalar) const {
        Tensor result = *this;
        result *= scalar;
        return result;
    }
    Tensor& operator*=(const double& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    std::string to_string() const { return "Tensor{" + to_string_recursive(0, 0) + "}"; }

    std::string to_string_recursive(size_t dim, int index) const {
        if (dim == shape.size()) {
            return std::to_string(data[index]);
        }
        std::string result = "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += "\n" + std::string(dim + 1, ' ') +
                      to_string_recursive(dim + 1, index * shape[dim] + i);
        }
        result += "\n" + std::string(dim, ' ') + "]";
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.to_string();
        return os;
    }

   private:
    size_t get_index(std::vector<size_t> index) const {
        size_t result = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            result += index[i];
            if (i < shape.size() - 1) {
                result *= shape[i + 1];
            }
        }
        return result;
    }
};

#endif  // TENSOR_H
