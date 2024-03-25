cmake_minimum_required(VERSION 3.27)
project(gpt_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(gpt_cpp
        singleNN.cpp
        singleNN.h)
