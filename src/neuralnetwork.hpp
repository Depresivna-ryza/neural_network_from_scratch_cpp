#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cassert>
#include <vector>

#include "miscellaneous.hpp"

using namespace std;

struct NeuralNetwork {
    vector<size_t> topology;
    vector<vector<double>> weights;
    vector<vector<double>> biases;

    vector<vector<double>> neuron_values;      // neuron outputs
    vector<vector<double>> neuron_potentials;  // neuron potentials
    vector<vector<double>> neuron_gradients;   // derivative of error with respect to neuron output
    vector<vector<double>> weight_gradients;   // derivative of error with respect to weight
    vector<vector<double>> bias_gradients;     // derivative of error with respect to bias

   public:
    NeuralNetwork(vector<size_t> t) : topology{t} {
        // randomly initialize weights and biases
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            vector<double> layer_weights;
            vector<double> layer_biases;
            for (size_t j = 0; j < topology[layer] * topology[layer + 1]; j++) {
                layer_weights.push_back(normal_he(topology[layer]));
            }
            for (size_t j = 0; j < topology[layer + 1]; j++) {
                layer_biases.push_back(normal_he(topology[layer]));
            }
            weights.push_back(layer_weights);
            biases.push_back(layer_biases);
        }

        // initialize neuron values
        for (size_t layer = 0; layer < topology.size(); layer++) {
            vector<double> layer_values;
            for (size_t j = 0; j < topology[layer]; j++) {
                layer_values.push_back(0);
            }
            neuron_values.push_back(layer_values);
        }

        // initialize neuron potentials
        for (size_t layer = 0; layer < topology.size(); layer++) {
            vector<double> layer_potentials;
            for (size_t j = 0; j < topology[layer]; j++) {
                layer_potentials.push_back(0);
            }
            neuron_potentials.push_back(layer_potentials);
        }

        // initialize neuron gradients
        for (size_t layer = 0; layer < topology.size(); layer++) {
            vector<double> layer_gradients;
            for (size_t j = 0; j < topology[layer]; j++) {
                layer_gradients.push_back(0);
            }
            neuron_gradients.push_back(layer_gradients);
        }

        // initialize weight gradients
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            vector<double> layer_gradients;
            for (size_t j = 0; j < topology[layer] * topology[layer + 1]; j++) {
                layer_gradients.push_back(0);
            }
            weight_gradients.push_back(layer_gradients);
        }

        // Initialize bias gradients
        for (size_t layer = 0; layer < topology.size() - 1; layer++) {
            vector<double> layer_gradients(topology[layer + 1], 0.0);
            bias_gradients.push_back(layer_gradients);
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
                neuron_potentials[layer][i] = sum;

                if (layer == topology.size() - 1)
                    neuron_values[layer][i] = sum;  // Linear activation for the output layer
                else {
                    neuron_values[layer][i] = relu(sum);  // ReLU for hidden layers
                }
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

    void reset_bias_gradients() {
        for (auto& layer_gradients : bias_gradients) {
            fill(layer_gradients.begin(), layer_gradients.end(), 0.0);
        }
    }

    void epoch(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double learning_rate) {
        assert(inputs.size() == targets.size());

        double total_error = 0;

        // Reset weight gradients to zero at the start of each epoch
        reset_weight_gradients();
        reset_bias_gradients();

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

        cout << "Average error for this epoch: " << total_error / inputs.size() << endl;
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
                // Update each bias by subtracting the learning rate multiplied by the accumulated gradient
                biases[layer][i] -= learning_rate * bias_gradients[layer][i];
            }
        }
    }

    vector<double> predict(const vector<double>& input) {
        feed_forward(input);
        return neuron_values.back();
    }
};

#endif // NEURALNETWORK_H
