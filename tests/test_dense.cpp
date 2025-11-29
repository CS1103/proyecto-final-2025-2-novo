#include "../src/layers/Dense.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_dense_forward() {
    Dense layer(2, 3);
    
    // Set known weights for testing
    layer.weights.data = {{0.5, 0.3, 0.2}, {0.1, 0.4, 0.6}};
    layer.bias.data = {{0.1, 0.2, 0.3}};
    
    Matrix input(1, 2);
    input.data = {{1.0, 2.0}};
    
    Matrix output = layer.forward(input);
    
    // Expected: [1.0, 2.0] * [[0.5, 0.3, 0.2], [0.1, 0.4, 0.6]] + [0.1, 0.2, 0.3]
    // = [0.5+0.2, 0.3+0.8, 0.2+1.2] + [0.1, 0.2, 0.3]
    // = [0.7, 1.1, 1.4] + [0.1, 0.2, 0.3]
    // = [0.8, 1.3, 1.7]
    
    assert(output.rows == 1);
    assert(output.cols == 3);
    assert(std::abs(output.data[0][0] - 0.8) < 1e-6);
    assert(std::abs(output.data[0][1] - 1.3) < 1e-6);
    assert(std::abs(output.data[0][2] - 1.7) < 1e-6);
    
    std::cout << "[PASS] Dense forward pass test" << std::endl;
}

void test_dense_backward() {
    Dense layer(2, 3);
    
    // Set known weights
    layer.weights.data = {{0.5, 0.3, 0.2}, {0.1, 0.4, 0.6}};
    layer.bias.data = {{0.1, 0.2, 0.3}};
    
    Matrix input(1, 2);
    input.data = {{1.0, 2.0}};
    
    // Forward pass first
    Matrix output = layer.forward(input);
    
    // Backward pass with gradient
    Matrix output_gradient(1, 3);
    output_gradient.data = {{0.1, 0.2, 0.3}};
    
    double learning_rate = 0.01;
    Matrix input_gradient = layer.backward(output_gradient, learning_rate);
    
    // Check that input gradient has correct shape
    assert(input_gradient.rows == 1);
    assert(input_gradient.cols == 2);
    
    std::cout << "[PASS] Dense backward pass test" << std::endl;
}

void test_dense_dimensions() {
    Dense layer(5, 10);
    
    Matrix input(3, 5); // 3 samples, 5 features
    input.setRandom();
    
    Matrix output = layer.forward(input);
    
    assert(output.rows == 3);
    assert(output.cols == 10);
    
    std::cout << "[PASS] Dense dimensions test" << std::endl;
}

int main() {
    std::cout << "Running Dense layer tests..." << std::endl;
    
    test_dense_forward();
    test_dense_backward();
    test_dense_dimensions();
    
    std::cout << "\nAll Dense layer tests passed!" << std::endl;
    return 0;
}
