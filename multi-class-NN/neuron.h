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
    explicit Neuron(int inputSize, double biasValue=1.0);

    double activation(double x);
    double forward(const std::vector<double> &inputs);

    const std::vector<double>& getWeights() const;
    double getBias() const;
    void setWeights(const std::vector<double>& newWeights);
    void setBias(double newBias);

    // backprop-specific methods
    void updateWeights(double LR, double delta);
    static double derivativeA(double x);
    double getWeightedSum() const;
    double getOutput() const;
};


#endif //GPT_CPP_NEURON_H
