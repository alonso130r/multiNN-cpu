//
// Created by Vijay Goyal on 2024-03-26.
//

#include "neuron.h"

Neuron::Neuron(int inputSize, double biasValue) : bias(biasValue), lastInput(0.0), lastOutput(0.0) {
    // initialize weights at random
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < inputSize; ++i) {
        weights.push_back(distribution(gen));
    }
}

double Neuron::activation(double x) {
    return std::max(0.0, x);
}

double Neuron::forward(const std::vector<double> &inputs) {
    // ensure input size matches weight size
    if (inputs.size() != weights.size()) {
        throw std::runtime_error("Input and weight size do not match.");
    }

    // weighted sum of inputs, bias
    double weightedSum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
    return activation(weightedSum);
}

void Neuron::updateWeights(double LR, double delta) {
    for (double &weight : weights) {
        weight -= LR * delta * lastInput; // gradient descent
    }
    bias -= LR * delta;
}

double Neuron::derivativeA(double x) {
    // ReLU is f(x) = max(0,x), so the derivative is piecewise
    if (x <= 0.0) {
        return 0.0;
    } else {
        return 1.0;
    }
}

const std::vector<double> &Neuron::getWeights() const {
    return weights;
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(double newBias) {
    bias = newBias;
}

void Neuron::setWeights(const std::vector<double> &newWeights) {
    weights = newWeights;
}

double Neuron::getWeightedSum() const {return lastInput;}
double Neuron::getOutput() const {return lastOutput;}