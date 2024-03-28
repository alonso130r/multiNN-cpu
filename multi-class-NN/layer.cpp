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

void Layer::updateWeights(const std::vector<double> &deltas, double LR) {
    assert(deltas.size() == neurons.size()); // sanity check 😭
    for (size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].updateWeights(LR, deltas[i]);
    }
}

std::vector<double>
Layer::backpropagate(const std::vector<double> &nextDeltas, const std::vector<std::vector<double>> &nextWeights) {
    std::vector<double> deltas(neurons.size());

    for (size_t i = 0; i < neurons.size(); ++i) {
        double delta = 0.0;
        for (size_t j = 0; j < nextDeltas.size(); ++j) {
            delta += nextDeltas[j] * nextWeights[j][i];
        }
        delta *= Neuron::derivativeA(neurons[i].getWeightedSum());
        deltas[i] = delta;
    }

    return deltas;
}