//
// Created by Vijay Goyal on 2024-03-26.
//

#include <iostream>
#include "neuron.h"

Neuron::Neuron(int inputSize, double biasValue) : bias(biasValue), lastInput(0.0), lastOutput(0.0) {
    // initialize weights at random
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < inputSize; ++i) {
        weights.push_back(distribution(gen));
        m_weights.push_back(0.0);
        v_weights.push_back(0.0);
        gradWeights.push_back(0.0);
    }

    std::cout << "Neuron created with " << weights.size() << " weights." << std::endl;
}

double Neuron::activation(double x) {
    return std::max(0.0, x);
}

double Neuron::forward(const std::vector<double> &inputs) {
    // ensure input size matches weight size
    if (inputs.size() != weights.size()) {
        std::cout << "Mismatch: " << inputs.size() << " inputs and " << weights.size() << " weights." << std::endl;
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

void Neuron::updateAdamW(double lr, double beta1, double beta2, double epsilon, double weightDecay, int t) {
    double beta1_t = pow(beta1, t);
    double beta2_t = pow(beta2, t);

    for (size_t i = 0; i < weights.size(); ++i) {
        // Update m and v with gradients, not weights
        m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * gradWeights[i];
        v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * gradWeights[i] * gradWeights[i];

        double m_hat = m_weights[i] / (1 - beta1_t);
        double v_hat = v_weights[i] / (1 - beta2_t);

        // Include weight decay in the update step
        weights[i] -= lr * m_hat / (sqrt(v_hat) + epsilon) + lr * weightDecay * weights[i];
    }

    // Update for bias using its gradient
    m_bias = beta1 * m_bias + (1 - beta1) * gradBias;
    v_bias = beta2 * v_bias + (1 - beta2) * gradBias * gradBias;

    double m_hat_bias = m_bias / (1 - beta1_t);
    double v_hat_bias = v_bias / (1 - beta2_t);

    bias -= lr * m_hat_bias / (sqrt(v_hat_bias) + epsilon); // No weight decay on bias
}

double Neuron::derivativeA(double x) {
    // ReLU is f(x) = max(0,x), so the derivative is piecewise
    if (x <= 0.0) {
        return 0.0;
    } else {
        return 1.0;
    }
}

void Neuron::setGradient(const std::vector<double> &gradients, double gradientBias) {
    gradWeights = gradients;
    gradBias = gradientBias;
}

const std::vector<double> &Neuron::getWeights() const {
    return weights;
}
void Neuron::setWeights(const std::vector<double> &newWeights) {
    weights = newWeights;
}

std::vector<double> Neuron::getMWeights() const {
    return m_weights;
}
void Neuron::setMWeights(const std::vector<double> &newMWeights) {
    m_weights = newMWeights;
}

std::vector<double> Neuron::getVWeights() const {
    return v_weights;
}
void Neuron::setVWeights(const std::vector<double> &newVWeights) {
    v_weights = newVWeights;
}

double Neuron::getBias() const {
    return bias;
}
void Neuron::setBias(double newBias) {
    bias = newBias;
}

double Neuron::getMBias() const {
    return m_bias;
}
void Neuron::setMBias(double newMBias) {
    m_bias = newMBias;
}

double Neuron::getVBias() const {
    return v_bias;
}
void Neuron::setVBias(double newVBias) {
    v_bias = newVBias;
}

double Neuron::getWeightedSum() const {return lastInput;}
double Neuron::getOutput() const {return lastOutput;}
