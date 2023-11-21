#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cassert>
#include <chrono>
#include <vector>

#include "matrix.hpp"
#include "miscellaneous.hpp"

using namespace std;

struct NeuralNetwork {
    double weight_decay;
    double momentum_factor;

    vector<size_t> topology;
    vector<Matrix> weights;
    vector<Matrix> biases;

    vector<Matrix> neuron_values;              // neuron outputs
    vector<Matrix> neuron_potentials;          // neuron potentials
    vector<Matrix> neuron_gradients;           // derivative of error with respect to neuron output
    vector<Matrix> weight_gradients;           // derivative of error with respect to weight
    vector<Matrix> bias_gradients;             // derivative of error with respect to bias

   public:
    NeuralNetwork(vector<size_t> t, double wd = 0, double mf = 0, bool d = false)
        : weight_decay{wd}, momentum_factor{mf}, topology{t} {
        // randomly initialize weights and biases
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            weights.push_back(Matrix::normal_he_create(topology[layer + 1], topology[layer], topology[layer]));
            biases.push_back(Matrix::normal_he_create(topology[layer + 1], 1, topology[layer]));
        }

        // initialize neuron values
        for (size_t layer = 0; layer < topology.size(); layer++) {
            neuron_values.push_back(Matrix::zero_create(topology[layer], 1));
        }

        // initialize neuron potentials
        for (size_t layer = 0; layer < topology.size(); layer++) {
            neuron_potentials.push_back(Matrix::zero_create(topology[layer], 1));
        }

        // initialize neuron gradients
        for (size_t layer = 0; layer < topology.size(); layer++) {
            neuron_gradients.push_back(Matrix::zero_create(topology[layer], 1));
        }

        // initialize weight gradients
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            weight_gradients.push_back(Matrix::zero_create(topology[layer + 1], topology[layer]));
        }
        // Initialize bias gradients
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            bias_gradients.push_back(Matrix::zero_create(topology[layer + 1], 1));
        }
    }

    double relu(double x) { return x > 0 ? x : 0; }

    double relu_derivative(double x) { return x > 0 ? 1 : 0; }

    vector<double> softmax(const vector<double>& x) {
        vector<double> result(x.size());
        double sum_exp = 0;

        for (auto value : x) {
            sum_exp += exp(value);
        }

        for (size_t i = 0; i < x.size(); i++) {
            result[i] = exp(x[i]) / sum_exp;
        }

        return result;
    }

    void feed_forward(const vector<double>& input, bool inference = false) {
        assert(input.size() == topology[0]);
        neuron_values[0] = input;

        for (size_t layer = 1; layer < topology.size(); ++layer) {
            for (size_t i = 0; i < topology[layer]; ++i) {
                double sum = 0;
                for (size_t j = 0; j < topology[layer - 1]; ++j) {
                    sum += neuron_values[layer - 1][j] * weights[layer - 1][j * topology[layer] + i];
                }
                sum += biases[layer - 1][i];
                neuron_potentials[layer][i] = sum;

                if (layer == topology.size() - 1)
                    neuron_values[layer][i] = sum;  // Linear activation for the output layer
                else {
                    neuron_values[layer][i] = relu(sum);  // ReLU for hidden layers
                }
            }
        }

        // Apply softmax to the output layer
        neuron_values.back() = softmax(neuron_values.back());
    }

    void back_propagate(const vector<double>& target) {
        assert(target.size() == topology.back());

        // Gradient for output layer
        for (size_t i = 0; i < topology.back(); i++) {
            neuron_gradients.back()[i] = neuron_values.back()[i] - target[i];
        }

        // Backpropagation for hidden layers
        for (size_t layer = topology.size() - 2; layer > 0; --layer) {
            for (size_t i = 0; i < topology[layer]; ++i) {

                double gradient_sum = 0;
                for (size_t j = 0; j < topology[layer + 1]; ++j) {
                    if (layer == topology.size() - 2)
                        gradient_sum += neuron_gradients[layer + 1][j] * weights[layer][i * topology[layer + 1] + j];
                    else {
                        gradient_sum += neuron_gradients[layer + 1][j] *
                                        relu_derivative(neuron_potentials[layer + 1][j]) *
                                        weights[layer][i * topology[layer + 1] + j];
                    }
                }
                neuron_gradients[layer][i] = gradient_sum;
            }
        }
    }

    void update_weight_gradients() {
        // Iterate over all layers except the input layer
        for (size_t layer = 0; layer < topology.size() - 1; ++layer) {
            for (size_t i = 0; i < topology[layer]; ++i) {          // Iterate over neurons in the current layer
                for (size_t j = 0; j < topology[layer + 1]; ++j) {  // Iterate over neurons in the next layer
                    double gradient = 0;
                    if (layer == topology.size() - 2)
                        // For the output layer, the gradient is simply the neuron gradient multiplied by the input
                        gradient += neuron_gradients[layer + 1][j] * neuron_values[layer][i];
                    else {
                        // For hidden layers, the gradient is the neuron gradient multiplied by the input and the
                        // derivative of the activation function
                        gradient += neuron_gradients[layer + 1][j] * relu_derivative(neuron_potentials[layer + 1][j]) *
                                    neuron_values[layer][i];
                    }
                    // Update the weight gradient
                    weight_gradients[layer][i * topology[layer + 1] + j] += gradient;
                }
            }
        }

        // Update bias gradients
        for (size_t layer = 0; layer < topology.size() - 1; ++layer) {
            for (size_t i = 0; i < topology[layer + 1]; ++i) {
                if (layer == topology.size() - 2)
                    // For the output layer, the gradient is simply the neuron gradient
                    bias_gradients[layer][i] += neuron_gradients[layer + 1][i];
                else {
                    // For hidden layers, the gradient is the neuron gradient multiplied by the derivative of the
                    // activation function
                    bias_gradients[layer][i] +=
                        neuron_gradients[layer + 1][i] * relu_derivative(neuron_potentials[layer + 1][i]);
                }
            }
        }
    }

    void reset_weight_gradients() {
        for (auto& layer_gradients : weight_gradients) {
            fill(layer_gradients.begin(), layer_gradients.end(), 0.0);
        }
    }
    void reset_weight_gradients_momentum() {
        for (auto& layer_gradients : weight_gradients) {
            for (auto& gradient : layer_gradients) {
                gradient *= momentum_factor;
            }
        }
    }

    void reset_bias_gradients() {
        for (auto& layer_gradients : bias_gradients) {
            fill(layer_gradients.begin(), layer_gradients.end(), 0.0);
        }
    }

    void reset_bias_gradients_momentum() {
        for (auto& layer_gradients : bias_gradients) {
            for (auto& gradient : layer_gradients) {
                gradient *= momentum_factor;
            }
        }
    }

    void epoch(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double learning_rate) {
        assert(inputs.size() == targets.size());

        double total_error = 0;

        // Reset weight gradients to zero at the start of each epoch
        reset_weight_gradients_momentum();
        reset_bias_gradients_momentum();

        // Iterate over each input-target pair
        for (size_t i = 0; i < inputs.size(); ++i) {
            feed_forward(inputs[i]);
            back_propagate(targets[i]);
            update_weight_gradients();
            total_error += cross_entropy_loss(targets[i]);
        }

        // Update weights after accumulating gradients from all input-target pairs
        update_weights(learning_rate);
        update_biases(learning_rate);

        // cout << "Average error for this epoch: " << total_error / inputs.size() << endl;
    }

    double cross_entropy_loss(const vector<double>& target) {
        assert(target.size() == neuron_values.back().size());
        double loss = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            // Ensure the predicted value is not exactly 0 to avoid log(0)
            double predicted = max(neuron_values.back()[i], numeric_limits<double>::min());
            loss -= target[i] * log(predicted);
        }
        return loss / target.size();
    }

    double mean_squared_error(const vector<double>& target) {
        assert(target.size() == neuron_values.back().size());
        double loss = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            double error = neuron_values.back()[i] - target[i];
            loss += std::pow(error, 2);
        }
        return loss / target.size();
    }

    void update_weights(double learning_rate) {
        for (size_t layer = 0; layer < topology.size() - 1; ++layer) {
            for (size_t i = 0; i < topology[layer]; ++i) {
                for (size_t j = 0; j < topology[layer + 1]; ++j) {
                    weights[layer][i * topology[layer + 1] + j] *= (1 - weight_decay);
                    // Update each weight by subtracting the learning rate multiplied by the accumulated gradient
                    weights[layer][i * topology[layer + 1] + j] -=
                        learning_rate * weight_gradients[layer][i * topology[layer + 1] + j];
                }
            }
        }
    }

    void update_biases(double learning_rate) {
        for (size_t layer = 0; layer < topology.size() - 1; ++layer) {
            for (size_t i = 0; i < topology[layer + 1]; ++i) {
                biases[layer][i] *= (1 - weight_decay);
                // Update each bias by subtracting the learning rate multiplied by the accumulated gradient
                biases[layer][i] -= learning_rate * bias_gradients[layer][i];
            }
        }
    }

    vector<double> inference(const vector<double>& input) {
        feed_forward(input, true);
        return neuron_values.back();
    }

    auto inference(const vector<vector<double>>& input) {
        vector<vector<double>> predicted_labels;
        for (size_t i = 0; i < input.size(); ++i) {
            auto pred = inference(input[i]);
            predicted_labels.push_back(pred);
        }
        return predicted_labels;
    }

    void inference_and_output(string input_file, string output_file) {
        auto validate_vectors = read_csv(input_file);
        normalize_data(validate_vectors, 0, 255);

        auto predicted_validate = inference(validate_vectors);
        vector<double> predicted_validate_labels;
        for (auto& vec : predicted_validate) {
            predicted_validate_labels.push_back(argmax(vec));
        }

        vector_to_file(predicted_validate_labels, output_file);
    }
};

double run_network(int epochs = 1000,                        // Number of epochs
                   size_t batch_size = 198,                  // Batch size
                   double learning_rate = 0.000578,            // Learning rate
                   double momentum = 0.0000773781,                   // Momentum
                   double weight_decay = 0.000121,             // Weight decay
                   vector<size_t> hidden_layers = {48, 19},  // Topology of the network
                   size_t time_limit = 60 * 10 - 30,              // Time limit in seconds
                   bool verbose = true) {
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

    NeuralNetwork nn(topology, momentum, weight_decay);

    NeuralNetwork best_nn(nn);
    double best_score = 0;

    // Random engine for shuffling the data in SGD
    random_device rd;
    default_random_engine rng(rd());

    test_network(nn, test_vectors, test_labels, train_vectors, train_labels, to_string(0), verbose);

    auto const end = std::chrono::system_clock::now() + std::chrono::seconds(time_limit);

    for (int epoch = 0; epoch < epochs && std::chrono::system_clock::now() < end; ++epoch) {
        if (verbose) {
            cout << "Epoch " << (epoch + 1) << "/" << epochs << " ("
                 << (std::chrono::duration_cast<std::chrono::seconds>(end - std::chrono::system_clock::now()).count())
                 << " seconds left)" << endl;
        }

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
            test_network(nn, test_vectors, test_labels, train_vectors, train_labels, to_string(epoch + 1), verbose);
        if (accuracy > best_score) {
            best_score = accuracy;
            best_nn = nn;
        }
    }

    nn = best_nn;

    cout << "Training complete." << endl;

    // Run the network on the test data
    nn.inference_and_output("data/fashion_mnist_test_vectors.csv", "data/test_predictions.csv");
    nn.inference_and_output("data/fashion_mnist_train_vectors.csv", "data/train_predictions.csv");

    return best_score;
}

#endif // NEURALNETWORK_H
