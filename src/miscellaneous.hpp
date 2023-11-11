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

template <typename T>
void debug(std::string func_name, int line, std::vector<std::vector<T>> vec) {
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '";
    for (auto& m : vec) {
        for (auto& n : m) {
            std::cout << n << " ";
        }
        std::cout << "\n";
    }
    std::cout << "'\n";
}

double normal_he(double n) {
    std::normal_distribution<double> d(0, 2/n);
    std::random_device rd;
    std::mt19937 gen(rd());
    return d(gen);

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

std::tuple<vector<vector<double>>, vector<vector<double>>> create_xor(int dimension) {
    vector<vector<double>> train_vectors;
    vector<vector<double>> train_labels;
    for (int i = 0; i < std::pow(2, dimension); ++i) {
        vector<int> vec(dimension, 0);
        for (int j = 0; j < dimension; ++j) {
            vec[j] = (i >> j) & 1;
        }
        train_vectors.push_back(vector<double>(vec.begin(), vec.end()));
        train_labels.push_back({static_cast<double>(std::accumulate(vec.begin(), vec.end(), 0) % 2)});
    }
    return {train_vectors, train_labels};
}

auto split_to_train_and_test(vector<vector<double>>& input_data, vector<vector<double>>& output_data,
                             double train_ratio) {
    assert(input_data.size() == output_data.size());
    //shuffle the data
    vector<size_t> indices(input_data.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

    size_t train_size = static_cast<size_t>(input_data.size() * train_ratio);
    vector<vector<double>> train_vectors;
    vector<vector<double>> train_labels;
    vector<vector<double>> test_vectors;
    vector<vector<double>> test_labels;

    for (size_t i = 0; i < train_size; ++i) {
        train_vectors.push_back(input_data[indices[i]]);
        train_labels.push_back(output_data[indices[i]]);
    }

    for (size_t i = train_size; i < input_data.size(); ++i) {
        test_vectors.push_back(input_data[indices[i]]);
        test_labels.push_back(output_data[indices[i]]);
    }

    return std::make_tuple(train_vectors, train_labels, test_vectors, test_labels);
}

void test_network(auto nn, auto test_vectors, auto test_labels, auto train_vectors, auto train_labels, auto label) {
    auto predicted_test = nn.predict(test_vectors);
    auto predicted_train = nn.predict(train_vectors);

    size_t correct_test = 0;
    size_t correct_train = 0;

    for (size_t i = 0; i < test_vectors.size(); ++i) {
        if (argmax(predicted_test[i]) == argmax(test_labels[i])) {
            ++correct_test;
        }
    }

    for (size_t i = 0; i < train_vectors.size(); ++i) {
        if (argmax(predicted_train[i]) == argmax(train_labels[i])) {
            ++correct_train;
        }
    }

    cout << "Test accuracy: " << (static_cast<double>(correct_test) / test_vectors.size()) * 100 << "% ";
    cout << "Train accuracy: " << (static_cast<double>(correct_train) / train_vectors.size()) * 100 << "%" << endl;

    vector_to_file(predicted_test, "predicted_test_" + label + ".csv");
}


#endif // MISC_H
