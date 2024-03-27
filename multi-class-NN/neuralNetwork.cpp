//
// Created by Vijay Goyal on 2024-03-26.
//

#include "neuralNetwork.h"


void NeuralNetwork::addHiddenLayer(int numberOfNeurons, int inputSize) {
    layers.push_back(std::make_unique<HiddenLayer>(numberOfNeurons, inputSize));
}

void NeuralNetwork::addInputLayer(int numberOfNeurons, int inputSize) {
    layers.insert(layers.begin(), std::make_unique<InputLayer>(numberOfNeurons, inputSize));
}

void NeuralNetwork::addOutputLayer(int numberOfNeurons, int inputSize) {
    layers.push_back(std::make_unique<OutputLayer>(numberOfNeurons, inputSize));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &inputs) {
    std::vector<double> outputs = inputs;
    for (auto &layer : layers) {
        outputs = layer ->forward(outputs);
    }
    return outputs;
}