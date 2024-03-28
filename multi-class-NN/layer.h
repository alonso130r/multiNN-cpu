//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_LAYER_H
#define GPT_CPP_LAYER_H

#include "neuron.h"
#include <cassert>


class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int numberOfNeurons, int inputSize);
    virtual std::vector<double> forward(const std::vector<double> &inputs);
    std::vector<std::vector<double>> getWeights() const;
    std::vector<double> getBiases() const;
    std::vector<Neuron> getNeurons() const;

    // backprop methods
    std::vector<double> computeOutputDeltas(const std::vector<double> &expectedOutputs);
    void updateWeights(const std::vector<double> &deltas, double LR);
    std::vector<double> backpropagate(const std::vector<double> &nextDeltas, const std::vector<std::vector<double>> &nextWeights);
};


#endif //GPT_CPP_LAYER_H
