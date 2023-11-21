#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "miscellaneous.hpp"
#include "neuralnetwork.hpp"

mutex mtx;

using namespace std;

template <typename T>
vector<T> randomize_hyperarameter(T value) {
    T a = value;
    T b = value;
    vector<T> result;
    result.push_back(value);
    for (int i = 0; i < 10; ++i) {
        result.push_back(a *= 0.95);
        result.push_back(b *= 1.05);
    }
    return result;
}

template <typename T>
vector<vector<T>> randomize_hyperarameter(vector<T> value) {
    vector<vector<T>> values;
    for (auto& v : value) {
        values.push_back(randomize_hyperarameter(v));
    }
    vector<vector<T>> result;
    for (int i = 0; i < values[0].size(); ++i) {
        vector<T> temp;
        for (auto& v : values) {
            temp.push_back(v[i]);
        }
        result.push_back(temp);
    }
    return result;
}

template <typename T>
T sample_hyperparameter(vector<T> values) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, values.size() - 1);
    return values[dis(gen)];
}

void output_hyperarameters(double accuracy, int epochs, size_t batch_size, double learning_rate, double momentum,
                           double weight_decay, size_t h1, size_t h2, bool use_dropout, auto out_file) {
    cout << "accuracy: " << accuracy << " epochs: " << epochs << " batch_size: " << batch_size
         << " learning_rate: " << learning_rate << " momentum: " << momentum << " weight_decay: " << weight_decay
         << " hidden_layer_1: " << h1 << " hidden_layer_2: " << h2 << " use_dropout: " << use_dropout << endl;

    // append at the end of file
    *out_file << accuracy << "," << epochs << "," << batch_size << "," << learning_rate << "," << momentum << ","
              << weight_decay << "," << h1 << "," << h2 << "," << use_dropout << endl;
}

void run_experiment(int epochs, size_t batch_size, double learning_rate, double momentum, double weight_decay,
                    size_t h1, size_t h2, bool use_dropout, size_t time_limit, ofstream& out_file) {
    double accuracy =
        run_network(epochs, batch_size, learning_rate, momentum, weight_decay, {h1, h2}, use_dropout, time_limit);

    // Synchronize access to the output file
    lock_guard<mutex> lock(mtx);
    output_hyperarameters(accuracy, epochs, batch_size, learning_rate, momentum, weight_decay, h1, h2, use_dropout,
                          &out_file);
    
}

int main() {
    // default hyperparameters
    int epochs = 1000;              // Number of epochs
    size_t batch_size = 150;        // Batch size
    double learning_rate = 0.0005;  // Learning rate
    double momentum = 0.00;         // Momentum
    double weight_decay = 0.0000;   // Weight decay
    size_t hidden_layer_1 = 80;     // Topology of the network
    size_t hidden_layer_2 = 20;     // Topology of the network
    bool use_dropout = false;       // Use dropout
    size_t time_limit = 1;          // Time limit in seconds

    auto h_batch_size = randomize_hyperarameter(batch_size);
    auto h_learning_rate = randomize_hyperarameter(learning_rate);
    auto h_momentum = randomize_hyperarameter(momentum);
    auto h_weight_decay = randomize_hyperarameter(weight_decay);
    auto h_hidden_layer_1 = randomize_hyperarameter(hidden_layer_1);
    auto h_hidden_layer_2 = randomize_hyperarameter(hidden_layer_2);

    auto out_file = ofstream("hyperparam_search.csv", ios::app);
    // write header
    out_file
        << "accuracy,epochs,batch_size,learning_rate,momentum,weight_decay,hidden_layer_1,hidden_layer_2,use_dropout"
        << endl;

    while (true) {
        vector<thread> threads;
        size_t num_threads = 2;

        for (size_t i = 0; i < num_threads; ++i) {
            auto n_batch_size = sample_hyperparameter(h_batch_size);
            auto n_learning_rate = sample_hyperparameter(h_learning_rate);
            auto n_momentum = sample_hyperparameter(h_momentum);
            auto n_weight_decay = sample_hyperparameter(h_weight_decay);
            auto n_hidden_layer_1 = sample_hyperparameter(h_hidden_layer_1);
            auto n_hidden_layer_2 = sample_hyperparameter(h_hidden_layer_2);

            threads.emplace_back(run_experiment, epochs, n_batch_size, n_learning_rate, n_momentum, n_weight_decay,
                                 n_hidden_layer_1, n_hidden_layer_2, use_dropout, time_limit, ref(out_file));
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    return 0;
}