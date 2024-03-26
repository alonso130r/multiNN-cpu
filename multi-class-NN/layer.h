//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_LAYER_H
#define GPT_CPP_LAYER_H

#include "neuron.h"



class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int numberOfNeurons, int inputSize);
    std::vector<double> forward(const std::vector<double> &inputs);
};


#endif //GPT_CPP_LAYER_H
