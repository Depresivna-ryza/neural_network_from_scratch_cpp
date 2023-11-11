#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>


#define DEBUG(message) debug(__FUNCTION__, __LINE__, message)

using namespace std;

void debug(std::string func_name, int line, auto message) {
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '"
              << message << "'\n";
}

template <typename T>
void debug(std::string func_name, int line, std::vector<T> vec) {
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '";
    for (auto& m : vec) {
        std::cout << m << " ";
    }
    std::cout << "'\n";
}

double random_gaussian() {
    std::normal_distribution<double> d(0, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    return d(gen);
}

vector<size_t> shuffle_indices(size_t n) {
    vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    return indices;
}

template <typename T>
vector<T> shuffle_data(const vector<T>& data, const vector<size_t>& indices) {
    vector<T> shuffled(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        shuffled[i] = data[indices[i]];
    }
    return shuffled;
}

vector<double> parse_csv_line(const string& line) {
    vector<double> result;
    stringstream lineStream(line);
    string cell;

    while (getline(lineStream, cell, ',')) {
        result.push_back(stod(cell));
    }

    return result;
}

vector<double> label_to_one_hot_vector(const vector<double>& label) {
    vector<double> result(10, 0);
    result[label[0]] = 1;
    return result;
}

vector<vector<double>> label_to_one_hot_vector(const vector<vector<double>>& labels) {
    vector<vector<double>> result;
    for (auto& label : labels) {
        result.push_back(label_to_one_hot_vector(label));
    }
    return result;
}

vector<vector<double>> read_csv(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    assert(file.is_open());
    string line;

    while (getline(file, line)) {
        data.push_back(parse_csv_line(line));
    }

    return data;
}

void vector_to_file(const vector<double>& vec, const string& filename) {
    ofstream file(filename);
    assert(file.is_open());

    for (auto& value : vec) {
        file << value << "\n";
    }
}

void vector_to_file(const vector<vector<double>>& vec, const string& filename) {
    ofstream file(filename);
    assert(file.is_open());

    for (auto& row : vec) {
        for (auto& value : row) {
            file << value << ",";
        }
        file << "\n";
    }
}

void normalize_data(vector<vector<double>>& data, double min, double max) {
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            data[i][j] = (data[i][j] - min) / (max - min);
        }
    }
}

size_t argmax(const vector<double>& vec) {
    double max = vec[0];
    size_t max_index = 0;
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > max) {
            max = vec[i];
            max_index = i;
        }
    }
    return max_index;
}
#endif // MISC_H
