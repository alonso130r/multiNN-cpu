cmake_minimum_required(VERSION 3.27)
project(gpt_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(gpt_cpp
        singleNN.cpp
        singleNN.h
        multi-class-NN/neuron.h
        multi-class-NN/neuron.cpp
        multi-class-NN/layer.h
        multi-class-NN/layer.cpp
        multi-class-NN/layerWrappers.h
        multi-class-NN/layerWrappers.cpp)
