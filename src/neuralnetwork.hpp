#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cassert>
#include <vector>

#include "misc.hpp"

using namespace std;

class NeuralNetwork
{
    vector<size_t> topology;
    vector<vector<double>> weights;
    vector<vector<double>> biases;

    vector<vector<double>> neuron_values;
    vector<vector<double>> neuron_gradients;

   public:
    NeuralNetwork(vector<size_t> t) : topology{t} {
        // randomly initialize weights and biases
        for (size_t i = 0; i < topology.size() - 1; i++) {
            vector<double> layer_weights;
            vector<double> layer_biases;
            for (size_t j = 0; j < topology[i] * topology[i + 1]; j++) {
                layer_weights.push_back(random_gaussian());
            }
            for (size_t j = 0; j < topology[i + 1]; j++) {
                layer_biases.push_back(random_gaussian());
            }
            weights.push_back(layer_weights);
            biases.push_back(layer_biases);
        }

        // initialize neuron values
        for (size_t i = 0; i < topology.size(); i++) {
            vector<double> layer_values;
            for (size_t j = 0; j < topology[i]; j++) {
                layer_values.push_back(0);
            }
            neuron_values.push_back(layer_values);
        }

        // initialize neuron gradients
        for (size_t i = 0; i < topology.size(); i++) {
            vector<double> layer_gradients;
            for (size_t j = 0; j < topology[i]; j++) {
                layer_gradients.push_back(0);
            }
            neuron_gradients.push_back(layer_gradients);
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

    void feed_forward(const vector<double>& input) {
        assert(input.size() == topology[0]);
        neuron_values[0] = input;

        for (size_t layer = 1; layer < topology.size(); ++layer) {
            for (size_t i = 0; i < topology[layer]; ++i) {
                double sum = 0;
                for (size_t j = 0; j < topology[layer - 1]; ++j) {
                    sum += neuron_values[layer - 1][j] * weights[layer - 1][j * topology[layer] + i];
                }
                sum += biases[layer - 1][i];

                // Apply ReLU for hidden layers and softmax for the output layer
                if (layer == topology.size() - 1) {  // Check if it's the last hidden layer
                    neuron_values[layer][i] = sum;   // Output before softmax
                } else {
                    neuron_values[layer][i] = relu(sum);  // ReLU for hidden layers
                }
            }

            // Apply softmax at the output layer
            if (layer == topology.size() - 1) {
                neuron_values[layer] = softmax(neuron_values[layer]);
            }
        }
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
                    gradient_sum += neuron_gradients[layer + 1][j] * weights[layer][i * topology[layer + 1] + j] *
                                    relu_derivative(neuron_values[layer + 1][j]);
                }
                neuron_gradients[layer][i] = gradient_sum;
            }
        }
    }
};

#endif // NEURALNETWORK_H
