//
// Created by Vijay Goyal on 2024-03-24.
//

#include <singleNN.h>
#include <ctime>
using namespace std;

// implement sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// implement sigmoid derivative
double sigmoid_derivative(double x) {
    return (double) 1.0 / (1.0 - sigmoid(x));
}

// single layer NN constructor
singleNN::singleNN(const std::vector<int> &sizes, double lr) {
    layer_sizes = sizes;
    learn_rate = lr;

    // initialize weights and biases
    for (int i = 0; i < sizes.size() - 1; i++) {
        vector<double> layer_weights(sizes[i] * sizes[i+1]);
        auto layer_bias = (double) rand() / RAND_MAX;

        // initialize weights randomly
        for (auto &weight : layer_weights) {
            weight = (double) rand() / RAND_MAX;
        }

        weights.push_back(layer_weights);
        biases.push_back(layer_bias);
    }
}

// forward propagation
std::vector<double> singleNN::forward(const std::vector<double>& input) {
    vector<double> activation = input;
    outputs.resize(2); // store activations for both layers

    // for each layer
    for (int i = 0; i < layer_sizes.size() - 1; i++) {
        vector<double> next_activation(layer_sizes[i+1]);

        // for each neuron in layer
        for (int j = 0; j < layer_sizes[i+1]; j++) {
            double z = 0.0;

            // for each input to neuron
            for (int k = 0; k < layer_sizes[i]; k++) {
                z += activation[k] * weights[i][j * layer_sizes[i] + k];
            }

            z += biases[i];
            next_activation[j] = sigmoid(z);
        }
        // save activation for use in backprop
        if (i == 0) {
            outputs[0] = next_activation;
        }

        activation = next_activation;
    }

    return activation;
}

void singleNN::backpropagate(const vector<double> &inputs, const vector<double> &expected) {
    this->forward(inputs);

    // calculate output layer error (delta)
    vector<double> output_errors(outputs[1].size());
    for (size_t i = 0; i < outputs[1].size(); ++i) {
        output_errors[i] = (expected[i] - outputs[1][i] * sigmoid_derivative(outputs[1][i]));
    }

    // calculate hidden layer error
    vector<double> hidden_layer_error(outputs[0].size(), 0.0);
    for (size_t i = 0; i < layer_sizes[2]; ++i) {
        for (size_t j = 0; j < layer_sizes[1]; ++j) {
            hidden_layer_error[j] += output_errors[i] * weights[1][i * layer_sizes[1] + j];
        }
    }

    for (size_t i = 0; i < hidden_layer_error.size(); ++i) {
        hidden_layer_error[i] *= sigmoid_derivative(outputs[0][i]);
    }

    // update weights for hidden-output layer
    for (size_t i = 0; i < layer_sizes[2]; ++i) {
        for (size_t j = 0; j < layer_sizes[1]; ++j) {
            weights[1][i * layer_sizes[1] + j] += learn_rate * output_errors[i] * outputs[0][j];
        }
    }

    // update weights for input-hidden layer
    for (size_t i = 0; i < layer_sizes[1]; ++i) {
        for (size_t j = 0; j < layer_sizes[0]; ++j) {
            weights[0][i * layer_sizes[0] + j] += learn_rate * hidden_layer_error[i] * inputs[j];
        }
    }

    // update biases for hidden layer
    for (size_t i = 0; i < layer_sizes[1]; ++i) {
        biases[i] += learn_rate * output_errors[i];
    }

    // update biases for output layer
    for (size_t i = 0; i < layer_sizes[2]; ++i) {
        biases[layer_sizes[1] + i] += learn_rate * output_errors[i];
    }
}

// mean squared error loss
double mse_loss(const vector<double>& outputs, const vector<double>& expected) {
    double sum = 0.0;
    for(size_t i = 0; i < outputs.size(); i++) {
        sum += pow(outputs[i] - expected[i], 2);
    }
    return sum / (double)outputs.size();
}