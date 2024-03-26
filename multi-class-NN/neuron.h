//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_NEURON_H
#define GPT_CPP_NEURON_H

#include <vector>
#include <cmath>
#include <numeric>
#include <random>


class Neuron {
private:
    std::vector<double> weights;
    double bias;

public:
    Neuron(int inputSize, double biasValue=1.0);
    double activation(double x);
    double forward(const std::vector<double> &inputs);
};


#endif //GPT_CPP_NEURON_H
