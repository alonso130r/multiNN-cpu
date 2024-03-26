//
// Created by Vijay Goyal on 2024-03-26.
//

#ifndef GPT_CPP_LAYERWRAPPERS_H
#define GPT_CPP_LAYERWRAPPERS_H

#include "layer.h"

class HiddenLayer : public Layer {
private:
    std::vector<Neuron> neurons;

public:
    using Layer::Layer;
};

class OutputLayer : public Layer {
private:
    std::vector<Neuron> neurons;

public:
    using Layer::Layer;
};

class InputLayer : public Layer {
private:
    std::vector<Neuron> neurons;
    std::vector<double> calculateMeans(const std::vector<std::vector<double>> &inputs);
    std::vector<double> calculateSTDDev(const std::vector<std::vector<double>> &inputs, const std::vector<double> &means);
    std::vector<std::vector<double>> applyNormalization(const std::vector<std::vector<double>> &inputs,
                                                        const std::vector<double> &means,
                                                        const std::vector<double> stdDevs);

public:
    using Layer::Layer;
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &inputs);
};
#endif //GPT_CPP_LAYERWRAPPERS_H
