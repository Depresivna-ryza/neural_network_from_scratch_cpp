#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "miscellaneous.hpp"
#include "neuralnetwork.hpp"
#include "tensor.hpp"

using namespace std;

int main() {
    // Hyperparameters
    int epochs = 1000;                        // Number of epochs
    size_t batch_size = 150;                  // Batch size
    double learning_rate = 0.0005;            // Learning rate
    double momentum = 0.00;                 // Momentum
    double weight_decay = 0.0001;             // Weight decay
    vector<size_t> hidden_layers = {80, 20};  // Topology of the network
    bool use_dropout = false;                 // Use dropout
    size_t time_limit = 60 * 10;

    // XOR data
    // auto [input_vector, output_labels] = create_xor(10);
    // auto [train_vectors, train_labels, test_vectors, test_labels] = split_to_train_and_test(input_vector,
    // output_labels, 1);

    // Read the data
    auto data_vector = read_csv("data/fashion_mnist_train_vectors.csv");
    auto data_labels = label_to_one_hot_vector(read_csv("data/fashion_mnist_train_labels.csv"));

    normalize_data(data_vector, 0, 255);

    auto [train_vectors, train_labels, test_vectors, test_labels] =
        split_to_train_and_test(data_vector, data_labels, 0.8);

    // Create the neural network
    size_t input_size = train_vectors[0].size();
    size_t output_size = train_labels[0].size();

    vector<size_t> topology = {input_size};
    topology.insert(topology.end(), hidden_layers.begin(), hidden_layers.end());
    topology.push_back(output_size);

    NeuralNetwork nn(topology, momentum, weight_decay, use_dropout);

    NeuralNetwork best_nn(nn);
    double best_score = 0;

    // Random engine for shuffling
    random_device rd;
    default_random_engine rng(rd());

    test_network(nn, test_vectors, test_labels, train_vectors, train_labels, to_string(0));

    auto const end =std::chrono::system_clock::now() + std::chrono::seconds(time_limit);

    for (int epoch = 0; epoch < epochs && std::chrono::system_clock::now() < end; ++epoch) {
        cout << "Epoch " << (epoch + 1) << "/" << epochs << " ("
             << (std::chrono::duration_cast<std::chrono::seconds>(end - std::chrono::system_clock::now()).count())
             << " seconds left)" << endl;

        // Shuffle the data at the beginning of each epoch
        vector<size_t> indices(train_vectors.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);

        // Iterate over batches
        for (size_t batch_start = 0; batch_start < train_vectors.size(); batch_start += batch_size) {
            size_t batch_end = min(batch_start + batch_size, train_vectors.size());
            vector<vector<double>> batch_vectors;
            vector<vector<double>> batch_labels;

            for (size_t i = batch_start; i < batch_end; ++i) {
                batch_vectors.push_back(train_vectors[indices[i]]);
                batch_labels.push_back(train_labels[indices[i]]);
            }
            // Run training epoch on the current batch
            nn.epoch(batch_vectors, batch_labels, learning_rate);
        }

        // Test the network
        double accuracy =
            test_network(nn, test_vectors, test_labels, train_vectors, train_labels, to_string(epoch + 1));
        if (accuracy > best_score) {
            best_score = accuracy;
            best_nn = nn;
        }
    }

    nn = best_nn;

    cout << "Training complete." << endl;

    // Run the network on the test data

    auto validate_vectors = read_csv("data/fashion_mnist_test_vectors.csv");
    normalize_data(validate_vectors, 0, 255);

    auto predicted_validate = nn.predict(validate_vectors);
    vector<double> predicted_validate_labels;
    for (auto& vec : predicted_validate) {
        predicted_validate_labels.push_back(argmax(vec));
    }

    vector_to_file(predicted_validate_labels, "predicted_validate.csv");

    return 0;
}
