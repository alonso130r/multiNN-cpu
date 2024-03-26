//
// Created by Vijay Goyal on 2024-03-26.
//

#include "layerWrappers.h"

std::vector<double> InputLayer::calculateMeans(const std::vector<std::vector<double>> &inputs) {
    std::vector<double> means(inputs[0].size(), 0.0);
    for (const auto &input : inputs) {
        for (size_t i = 0; i < input.size(); ++i) {
            means[i] += input[i];
        }
    }
    for (double &mean : means) {
        mean /= (double) inputs.size();
    }
    return means;
}

std::vector<double> InputLayer::calculateSTDDev(const std::vector<std::vector<double>> &inputs, const std::vector<double> &means) {
    std::vector<double> stdDevs(means.size(), 0.0);
    for (const auto &input : inputs) {
        for (size_t i = 0; i < input.size(); ++i) {
            stdDevs[i] += std::pow(input[i] - means[i], 2);
        }
    }
    for (double &stdDev : stdDevs) {
        stdDev = std::sqrt(stdDev / (double)inputs.size());
    }
    return stdDevs;
}

std::vector<std::vector<double>> InputLayer::applyNormalization(const std::vector<std::vector<double>> &inputs,
                                                                const std::vector<double> &means,
                                                                const std::vector<double> stdDevs) {
    std::vector<std::vector<double>> normalizedInputs = inputs;
    for (auto &input : normalizedInputs) {
        for (size_t i = 0; i < input.size(); ++i) {
            if (stdDevs[i] != 0) {
                input[i] = (input[i] - means[i]) / stdDevs[i];
            }
        }
    }
    return normalizedInputs;
}

std::vector<std::vector<double>> InputLayer::normalize(const std::vector<std::vector<double>> &inputs) {
    std::vector<double> means = calculateMeans(inputs);
    std::vector<double> stdDevs = calculateSTDDev(inputs, means);

    return applyNormalization(inputs, means, stdDevs);
}