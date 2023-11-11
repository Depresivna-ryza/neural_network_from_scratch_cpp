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
    // Read the data
    // auto train_vectors = read_csv("../../data/fashion_mnist_train_vectors.csv");
    // auto train_labels = label_to_one_hot_vector(read_csv("../../data/fashion_mnist_train_labels.csv"));

    // auto test_vectors = read_csv("../../data/fashion_mnist_test_vectors.csv");
    // auto test_labels = label_to_one_hot_vector(read_csv("../../data/fashion_mnist_test_labels.csv"));

    // normalize_data(train_vectors, 0, 255);

    vector<vector<double>> train_vectors = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> train_labels = {{0, 1}, {1, 0}, {1, 0}, {0, 1}};

    // Create the neural network
    size_t input_size = train_vectors[0].size();
    size_t output_size = train_labels[0].size();

    vector<size_t> topology = {input_size, 2, output_size};
    NeuralNetwork nn(topology);

    // Train the network
    int epochs = 100;             // Number of epochs
    size_t batch_size = 500;      // Batch size
    double learning_rate = 0.01;  // Learning rate

    // Random engine for shuffling
    random_device rd;
    default_random_engine rng(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << (epoch + 1) << "/" << epochs << endl;

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
            string out_file = "output" + to_string(epoch) + ".txt";
            vector_to_file(nn.neuron_values.back(), out_file);
        }

        // test the network on the test set
        // vector<vector<double>> predictions;
        // double correct = 0;
        // for (size_t i = 0; i < test_vectors.size(); ++i) {
        //     auto pred = nn.predict(test_vectors[i]);
        //     predictions.push_back(pred);
        //     if (argmax(pred) == argmax(test_labels[i])) {
        //         ++correct;
        //     }
        // }
        // cout << "Accuracy: " << correct / test_vectors.size() << endl;

        // string out_file = "prediction" + to_string(epoch) + ".txt";
        // vector_to_file(predictions, out_file);
    }

    cout << "Training complete." << endl;

    return 0;
}
