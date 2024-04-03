//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_NEURALNETWORK_H
#define GPT_CPP_NEURALNETWORK_H

#include "layerWrappers.h"
#include <memory>
#include <iostream>
#include <fstream>


class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    double lr, beta1, beta2, epsilon, weightDecay;
    int lastSize = 0;

public:
    explicit NeuralNetwork(double LR, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double weightDecay = 0.01);

    // add hidden layer
    void addHiddenLayer(int numberOfNeurons, int inputSize);
    void addHiddenLayer(int numberOfNeurons);

    // add input layer
    void addInputLayer(int numberOfNeurons, int inputSize);

    // add output layer
    void addOutputLayer(int numberOfNeurons, int inputSize);

    // forward prop
    std::vector<double> forward(const std::vector<double> &inputs);

    // backprop/train methods
    void backpropagate(const std::vector<double> &expected);
    void train(const std::vector<std::vector<double>>& dataset, const std::vector<std::vector<double>>& labels, int epochs, const std::string &filename);

    // save/load weights, binary file
    void save(const std::string &filename);
    void load(const std::string &filename);
};


#endif //GPT_CPP_NEURALNETWORK_H
