#include "../src/layers/Activation.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_tanh_forward() {
    Tanh activation;
    
    Matrix input(2, 2);
    input.data = {{0.0, 1.0}, {-1.0, 2.0}};
    
    Matrix output = activation.forward(input);
    
    assert(output.rows == 2);
    assert(output.cols == 2);
    assert(std::abs(output.data[0][0] - std::tanh(0.0)) < 1e-6);
    assert(std::abs(output.data[0][1] - std::tanh(1.0)) < 1e-6);
    assert(std::abs(output.data[1][0] - std::tanh(-1.0)) < 1e-6);
    assert(std::abs(output.data[1][1] - std::tanh(2.0)) < 1e-6);
    
    std::cout << "[PASS] Tanh forward pass test" << std::endl;
}

void test_tanh_backward() {
    Tanh activation;
    
    Matrix input(1, 2);
    input.data = {{0.5, -0.5}};
    
    Matrix output = activation.forward(input);
    
    Matrix output_gradient(1, 2);
    output_gradient.data = {{1.0, 1.0}};
    
    Matrix input_gradient = activation.backward(output_gradient, 0.01);
    
    // Derivative of tanh(x) is 1 - tanh²(x)
    double tanh_05 = std::tanh(0.5);
    double tanh_m05 = std::tanh(-0.5);
    double expected_grad_0 = 1.0 * (1 - tanh_05 * tanh_05);
    double expected_grad_1 = 1.0 * (1 - tanh_m05 * tanh_m05);
    
    assert(std::abs(input_gradient.data[0][0] - expected_grad_0) < 1e-6);
    assert(std::abs(input_gradient.data[0][1] - expected_grad_1) < 1e-6);
    
    std::cout << "[PASS] Tanh backward pass test" << std::endl;
}

void test_sigmoid_forward() {
    Sigmoid activation;
    
    Matrix input(2, 2);
    input.data = {{0.0, 1.0}, {-1.0, 2.0}};
    
    Matrix output = activation.forward(input);
    
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    
    assert(output.rows == 2);
    assert(output.cols == 2);
    assert(std::abs(output.data[0][0] - sigmoid(0.0)) < 1e-6);
    assert(std::abs(output.data[0][1] - sigmoid(1.0)) < 1e-6);
    assert(std::abs(output.data[1][0] - sigmoid(-1.0)) < 1e-6);
    assert(std::abs(output.data[1][1] - sigmoid(2.0)) < 1e-6);
    
    std::cout << "[PASS] Sigmoid forward pass test" << std::endl;
}

void test_sigmoid_backward() {
    Sigmoid activation;
    
    Matrix input(1, 2);
    input.data = {{0.5, -0.5}};
    
    Matrix output = activation.forward(input);
    
    Matrix output_gradient(1, 2);
    output_gradient.data = {{1.0, 1.0}};
    
    Matrix input_gradient = activation.backward(output_gradient, 0.01);
    
    // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    double sig_05 = sigmoid(0.5);
    double sig_m05 = sigmoid(-0.5);
    double expected_grad_0 = 1.0 * sig_05 * (1 - sig_05);
    double expected_grad_1 = 1.0 * sig_m05 * (1 - sig_m05);
    
    assert(std::abs(input_gradient.data[0][0] - expected_grad_0) < 1e-6);
    assert(std::abs(input_gradient.data[0][1] - expected_grad_1) < 1e-6);
    
    std::cout << "[PASS] Sigmoid backward pass test" << std::endl;
}

int main() {
    std::cout << "Running Activation function tests..." << std::endl;
    
    test_tanh_forward();
    test_tanh_backward();
    test_sigmoid_forward();
    test_sigmoid_backward();
    
    std::cout << "\nAll Activation function tests passed!" << std::endl;
    return 0;
}
