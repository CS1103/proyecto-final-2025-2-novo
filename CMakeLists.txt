cmake_minimum_required(VERSION 3.18)
project(NeuralNetworkDemo)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

# Library sources (excluding main.cpp)
set(LIB_SOURCES
    src/Network.cpp
    src/layers/Dense.cpp
)

# Main executables
add_executable(neural_net_demo src/main.cpp ${LIB_SOURCES})
add_executable(neural_net_large_demo src/main_large.cpp ${LIB_SOURCES})

# Test executables
add_executable(test_matrix tests/test_matrix.cpp)
add_executable(test_dense tests/test_dense.cpp ${LIB_SOURCES})
add_executable(test_activation tests/test_activation.cpp)
add_executable(test_xor tests/test_xor.cpp ${LIB_SOURCES})
