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
    std::cout << "Loading data from " << filename << "..." <<std::endl;
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
            } catch (const std::out_of_range& e) {
                // Skip line if number passed is out of range for a double
                invalidData = true;
                break;
            }
        }

        if (!invalidData) {
            int label = static_cast<int>(featureVector.back());
            featureVector.pop_back();

            // Directly use the label in a vector
            std::vector<double> labelVector = {static_cast<double>(label)};

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

    std::cout << features.size() << "\n";
    std::cout << labels.size() << "\n";
    std::cout << labels[1].size() << std::endl;

    NeuralNetwork nn1(1e-2);
    nn1.addInputLayer((int)features[1].size(), (int)features[1].size());
    nn1.addHiddenLayer(512, 1000);
    nn1.addHiddenLayer(256, 512);
    nn1.addHiddenLayer(128, 256);
    nn1.addHiddenLayer(64, 128); // eval test
    nn1.addHiddenLayer(32, 64); // eval test
    //nn1.addHiddenLayer(16, 32); // eval test
    nn1.addOutputLayer(1, 32);

    InputLayer::normalize(features);

    std::string file = "/Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/params2.bin";
    nn1.train(features, labels, 1, file);
}