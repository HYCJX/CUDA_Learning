cmake_minimum_required(VERSION 3.16)
project(Desktop CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Desktop main.cu Helper.cu Helper.cuh)

set_target_properties(
        Desktop
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)