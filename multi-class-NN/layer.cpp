//
// Created by Vijay Goyal on 2024-03-26.
//

#include "layer.h"

// complete constructor
Layer::Layer(int numberOfNeurons, int inputSize) {
    for (int i = 0; i < numberOfNeurons; ++i) {
        neurons.emplace_back(inputSize);
    }
}

// complete forward pass
std::vector<double> Layer::forward(const std::vector<double> &inputs) {
    inputs_ = inputs;
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
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
    // use inputs_

    for (size_t i = 0; i < neurons.size(); ++i) {
        double delta = 0.0;
        for (size_t j = 0; j < nextDeltas.size(); ++j) {
            delta += nextDeltas[j] * nextWeights[j][i];
        }
        delta *= Neuron::derivativeA(neurons[i].getWeightedSum());
        deltas[i] = delta;

        // calculate gradients
        std::vector<double> weightGradients;
        weightGradients.reserve(inputs_.size());
        for (const auto& input : inputs_) {
            weightGradients.push_back(delta * input);
        }
        double biasGradient = delta;
        neurons[i].setGradient(weightGradients, biasGradient);
    }

    return deltas;
}

std::vector<double> Layer::computeOutputDeltas(const std::vector<double> &expectedOutputs) {
    std::vector<double> outputDeltas(neurons.size());

    for (size_t i = 0; i < neurons.size(); ++i) {
        double output = neurons[i].getOutput();
        double derivative = neurons[i].derivativeA(output);
        double error = output - expectedOutputs[i];
        outputDeltas[i] = error * derivative;
    }
    return outputDeltas;
}

std::vector<std::vector<double>> Layer::getWeights() const {
    std::vector<std::vector<double>> layerWeights;
    for (const auto& neuron : neurons) {
        layerWeights.push_back(neuron.getWeights());
    }
    return layerWeights;
}

std::vector<double> Layer::getBiases() const {
    std::vector<double> layerBiases;
    for (const auto& neuron : neurons) {
        layerBiases.push_back(neuron.getBias());
    }
    return layerBiases;
}

std::vector<Neuron> Layer::getNeurons() const {
    return neurons;
}