//
// Created by Vijay Goyal on 2024-03-28.
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//int main() {
//    std::ifstream file("/Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/dataset1/train_pre.csv");
//    std::string line;
//    int lineCount = 0;
//
//    if (!file.is_open()) {
//        std::cerr << "Error opening file" << std::endl;
//        return 1;
//    }
//
//    while (std::getline(file, line)) {
//        std::stringstream linestream(line);
//        std::string cell;
//        std::vector<double> values;
//        lineCount++;
//        bool isFirstCell = true;
//
//        try {
//            while (std::getline(linestream, cell, ',')) {
//                // Skip the last column based on a condition or counter if it's a label
//                if (isFirstCell) {
//                    isFirstCell = false;
//                    continue; // Assuming the first column is not part of the data to convert
//                }
//                values.push_back(std::stod(cell));
//            }
//        } catch (const std::invalid_argument& ia) {
//            std::cerr << "Invalid argument on line " << lineCount << ": " << ia.what() << std::endl;
//            // Optionally, print the problematic line
//            std::cerr << "Problematic line: " << line << std::endl;
//        }
//
//        // Handle the last cell separately if it's a label or something else
//    }
//
//    file.close();
//    return 0;
//}
//
