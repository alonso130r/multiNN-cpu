//
// Created by Vijay Goyal on 2024-03-26.
//

#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(double LR, double b1, double b2,
                             double e, double wD) : lr(LR), beta1(b1), beta2(b2), epsilon(e), weightDecay(wD) {}

void NeuralNetwork::addHiddenLayer(int numberOfNeurons, int inputSize) {
    lastSize = inputSize;
    layers.push_back(std::make_unique<HiddenLayer>(numberOfNeurons, lastSize));
}

void NeuralNetwork::addHiddenLayer(int numberOfNeurons) {
    layers.push_back(std::make_unique<HiddenLayer>(numberOfNeurons, lastSize));
    lastSize /= 2;
}

void NeuralNetwork::addInputLayer(int inputSize, int numberOfNeurons) {
    layers.insert(layers.begin(), std::make_unique<InputLayer>(numberOfNeurons, inputSize));
    lastSize = inputSize;
}

void NeuralNetwork::addOutputLayer(int numberOfNeurons, int inputSize) {
    layers.push_back(std::make_unique<OutputLayer>(numberOfNeurons, inputSize));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &inputs) {
    std::vector<double> outputs = inputs;
    for (auto &layer : layers) {
        outputs = layer -> forward(outputs);
    }
    return outputs;
}

/* BACKPROP/TRAIN WORK BEGINS HERE */

void NeuralNetwork::backpropagate(const std::vector<double> &expected) {
    std::vector<double> deltas;
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
            // output layer
            deltas = layers[i] ->computeOutputDeltas(expected);
        } else {
            // hidden layers
            deltas = layers[i] ->backpropagate(deltas, layers[i+1] ->getWeights());
        }
        //layers[i] ->updateWeights(deltas, lr);
    }
}

void
NeuralNetwork::train(const std::vector<std::vector<double>> &dataset, const std::vector<std::vector<double>> &labels,
                     int epochs, const std::string &filename) {
    std::cout << "Training begun..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            std::vector<double> outputs = forward(dataset[i]);
            std::vector<double> probabilities = softmax(outputs);
            backpropagate(labels[i]);

            // calculate loss using cross-entropy
            loss += crossEntropyLoss(probabilities, labels[i]);

            // update weights w/ AdamW
            for (auto &layer : layers) {
                for (auto &neuron : layer->getNeurons()) {
                    neuron.updateAdamW(lr, beta1, beta2, epsilon, weightDecay, (int) i);
                }
            }

            if (i % 100 == 0) {
                std::cout << "Iteration " << i << ", loss of: " <<  loss / (double)i << std::endl;
            }
        }
        loss /= (double)dataset.size();
        printf("Epoch %d / %d, Loss: %0.5f\n", epoch + 1, epochs, loss);
    }
    save(filename);
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double> &inputs) {
    std::vector<double> exps(inputs.size());
    double maxInput = *std::max_element(inputs.begin(), inputs.end());
    double sum = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        exps[i] = std::exp(inputs[i] - maxInput); // Improve numerical stability
        sum += exps[i];
    }

    for (double& val : exps) {
        val /= sum;
    }

    return exps;
}

double NeuralNetwork::crossEntropyLoss(const std::vector<double>& outputs, const std::vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        loss -= labels[i] * std::log(outputs[i] + 1e-9); // Prevent log(0)
    }
    return -loss;
}

// save/load params from binary file

void NeuralNetwork::save(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (auto &neuron : layer->getNeurons()) {
            std::vector<double> weights = neuron.getWeights();
            std::vector<double> m_weights = neuron.getMWeights();
            std::vector<double> v_weights = neuron.getVWeights();
            size_t size = weights.size();
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            file.write(reinterpret_cast<const char*>(weights.data()), size * sizeof(double));
            file.write(reinterpret_cast<const char*>(m_weights.data()), size * sizeof(double));
            file.write(reinterpret_cast<const char*>(v_weights.data()), size * sizeof(double));
            double bias = neuron.getBias();
            double m_bias = neuron.getMBias();
            double v_bias = neuron.getVBias();
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
            file.write(reinterpret_cast<const char*>(&m_bias), sizeof(m_bias));
            file.write(reinterpret_cast<const char*>(&v_bias), sizeof(v_bias));
        }
    }

    file.close();
}

void NeuralNetwork::load(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    std::cout << "Loading weights from " << filename << "..." << std::endl;
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (auto &neuron : layer->getNeurons()) {
            size_t size;
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            std::vector<double> weights(size);
            std::vector<double> m_weights(size);
            std::vector<double> v_weights(size);
            file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(double));
            file.read(reinterpret_cast<char*>(m_weights.data()), size * sizeof(double));
            file.read(reinterpret_cast<char*>(v_weights.data()), size * sizeof(double));
            neuron.setWeights(weights);
            neuron.setMWeights(m_weights);
            neuron.setVWeights(v_weights);
            double bias;
            double m_bias;
            double v_bias;
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
            file.read(reinterpret_cast<char*>(&m_bias), sizeof(m_bias));
            file.read(reinterpret_cast<char*>(&v_bias), sizeof(v_bias));
            neuron.setBias(bias);
            neuron.setMBias(m_bias);
            neuron.setVBias(v_bias);
        }
    }

    file.close();
}

//void NeuralNetwork::save(const std::string &filename) {
//    std::ofstream file(filename, std::ios::binary);
//    if (!file.is_open()) {
//        std::cerr << "Failed to open file for writing: " << filename << std::endl;
//        return;
//    }
//
//    for (auto &layer : layers) {
//        for (auto &neuron : layer->getNeurons()) {
//            for (auto &weight : neuron.getWeights()) {
//                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
//            }
//            double bias = neuron.getBias();
//            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
//        }
//    }
//
//    file.close();
//}
//
//void NeuralNetwork::load(const std::string &filename) {
//    std::ifstream file(filename, std::ios::binary);
//    std::cout << "Loading weights from " << filename << "..." << std::endl;
//    if (!file.is_open()) {
//        std::cerr << "Failed to open file for reading: " << filename << std::endl;
//        return;
//    }
//
//    for (auto &layer : layers) {
//        for (auto &neuron : layer->getNeurons()) {
//            std::vector<double> weights(neuron.getWeights().size());
//            for (auto &weight : weights) {
//                file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
//            }
//            neuron.setWeights(weights);
//
//            double bias;
//            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
//            neuron.setBias(bias);
//        }
//    }
//
//    file.close();
//}