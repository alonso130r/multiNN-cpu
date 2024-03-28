//
// Created by Vijay Goyal on 2024-03-26.
//

#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(double LR) : lr(LR) {}

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

/* BACKPROP/TRAIN WORK BEGINS HERE */

void NeuralNetwork::backpropagate(const std::vector<double> &expected) {
    std::vector<double> deltas;
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
            // output layer
            deltas = layers[i] ->computeOutputDeltas(expected);
        } else {
            // hidden layers
            deltas = layers[i] ->backpropagate(deltas, layers[i+1] ->getWeights());
        }
        layers[i] ->updateWeights(deltas, lr);
    }
}

void
NeuralNetwork::train(const std::vector<std::vector<double>> &dataset, const std::vector<std::vector<double>> &labels,
                     int epochs, const std::string &filename) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            std::vector<double> outputs = forward(dataset[i]);
            backpropagate(labels[i]);

            // calculate loss using MSE
            for (size_t j = 0; j < labels[i].size(); ++j) {
                loss += std::pow(outputs[j] - labels[i][j], 2);
            }
        }
        loss /= (double)dataset.size();
        printf("Epoch %d / %d, Loss: %0.5f\n", epoch + 1, epochs, loss);
    }
    save(filename);
}

// save/load params from binary file

void NeuralNetwork::save(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (auto &neuron : layer->getNeurons()) {
            for (auto &weight : neuron.getWeights()) {
                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
            double bias = neuron.getBias();
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    file.close();
}

void NeuralNetwork::load(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (auto &neuron : layer->getNeurons()) {
            std::vector<double> weights(neuron.getWeights().size());
            for (auto &weight : weights) {
                file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
            neuron.setWeights(weights);

            double bias;
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
            neuron.setBias(bias);
        }
    }

    file.close();
}