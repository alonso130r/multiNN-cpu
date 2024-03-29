//
// Created by Vijay Goyal on 2024-03-27.
//

#include "neuralNetwork.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>


void loadCSV(const std::string &filename, std::vector<std::vector<double>> &features, std::vector<std::vector<double>> &labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::string cell;
        std::vector<double> featureVector;
        bool invalidData = false;

        while (std::getline(linestream, cell, ',')) {
            try {
                featureVector.push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                // Skip line if conversion fails due to non-numeric value
                invalidData = true;
                break;
            }
        }

        if (!invalidData) {
            int label = static_cast<int>(featureVector.back());
            featureVector.pop_back();

            // Convert label to 2D vector format
            std::vector<double> labelVector(2, 0.0);
            labelVector[label] = 1.0;

            features.push_back(featureVector);
            labels.push_back(labelVector);
        }
    }
}

int main() {
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;

    std::string f1 = "/Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/dataset1/train_pre.csv";

    loadCSV(f1, features, labels);

    NeuralNetwork nn1(1e-3);
    nn1.addInputLayer(1000, 1000);
    nn1.addHiddenLayer(500, 1000);
    nn1.addHiddenLayer(250, 500);
    nn1.addHiddenLayer(100, 250);
    nn1.addOutputLayer(1, 100);

    InputLayer::normalize(features);

    std::string file = "params1.bin";
    nn1.train(features, labels, 40, file);
}