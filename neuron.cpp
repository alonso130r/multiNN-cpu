//
// Created by Vijay Goyal on 2024-03-26.
//

#include "neuron.h"

Neuron::Neuron(int inputSize, double biasValue) : bias(biasValue){
    // initialize weights at random
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < inputSize; ++i) {
        weights.push_back(distribution(gen));
    }
}

double Neuron::activation(double x) {
    return std::tanh(x);
}

double Neuron::forward(const std::vector<double> &inputs) {
    // ensure input size matches weight size
    if (inputs.size() != weights.size()) {
        throw std::runtime_error("Input and weight size do not match.");
    }

    // weighted sum of inputs, bias
    double weightedSum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0) + bias;
    return activation(weightedSum);
}