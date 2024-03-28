//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_NEURON_H
#define GPT_CPP_NEURON_H

#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>


class Neuron {
private:
    std::vector<double> weights;
    double bias;
    double lastInput, lastOutput;

public:
    Neuron(int inputSize, double biasValue);

    double activation(double x);
    double forward(const std::vector<double> &inputs);

    // backprop-specific methods
    void updateWeights(double LR, double delta);
    static double derivativeA(double x);
    double getWeightedSum() const;
    double getOutput() const;
};


#endif //GPT_CPP_NEURON_H
