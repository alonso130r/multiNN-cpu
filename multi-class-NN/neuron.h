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
    std::vector<double> weights, m_weights, v_weights;
    std::vector<double> gradWeights;
    double bias, m_bias, v_bias;
    double gradBias;
    double lastInput, lastOutput;

public:
    explicit Neuron(int inputSize, double biasValue=1.0);

    double activation(double x);
    double forward(const std::vector<double> &inputs);

    const std::vector<double>& getWeights() const;
    void setWeights(const std::vector<double>& newWeights);
    std::vector<double> getMWeights() const;
    void setMWeights(const std::vector<double>& newMWeights);
    std::vector<double> getVWeights() const;
    void setVWeights(const std::vector<double>& newVWeights);
    double getBias() const;
    void setBias(double newBias);
    double getMBias() const;
    void setMBias(double newMBias);
    double getVBias() const;
    void setVBias(double newVBias);

    // backprop-specific methods
    void updateWeights(double LR, double delta);
    static double derivativeA(double x);
    double getWeightedSum() const;
    double getOutput() const;

    // for Adam optimizer (work in progress)
    void updateAdamW(double lr, double beta1, double beta2, double epsilon, double weightDecay, int t);
    void setGradient(const std::vector<double>& gradients, double gradientBias);
};


#endif //GPT_CPP_NEURON_H
