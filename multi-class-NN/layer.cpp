//
// Created by Vijay Goyal on 2024-03-26.
//

#include "layer.h"

// complete constructor
Layer::Layer(int numberOfNeurons, int inputSize) {
    for (int i = 0; i < inputSize; ++i) {
        neurons.emplace_back(inputSize);
    }
}

// complete forward pass
std::vector<double> Layer::forward(const std::vector<double> &inputs) {
    std::vector<double> outputs;
    for (auto &neuron: neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    return outputs;
}