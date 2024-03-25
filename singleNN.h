//
// Created by Vijay Goyal on 2024-03-24.
//

#ifndef GPT_CPP_SINGLENN_H
#define GPT_CPP_SINGLENN_H

#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

double sigmoid(double x);
double sigmoid_derivative(double x);

class singleNN {
public:
    vector<int> layer_sizes;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> outputs;

    // backpropagation storage
    double learn_rate;

    singleNN(const std::vector<int> &sizes, double lr);
    vector<double> forward(const vector<double> &input);
    void backpropagate(const vector<double> &inputs, const vector<double> &expected);
};

double mse_loss(const std::vector<double> &outputs, const std::vector<double> &expected);

#endif //GPT_CPP_SINGLENN_H
