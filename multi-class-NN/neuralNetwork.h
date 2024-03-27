//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_NEURALNETWORK_H
#define GPT_CPP_NEURALNETWORK_H

#include "layerWrappers.h"
#include <memory>


class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    NeuralNetwork() = default;

    // add hidden layer
    void addHiddenLayer(int numberOfNeurons, int inputSize);

    // add input layer
    void addInputLayer(int numberOfNeurons, int inputSize);

    // add output layer
    void addOutputLayer(int numberOfNeurons, int inputSize);

    // forward prop
    std::vector<double> forward(const std::vector<double> &inputs);
};


#endif //GPT_CPP_NEURALNETWORK_H
