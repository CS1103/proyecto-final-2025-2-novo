#include "Dense.h"

Dense::Dense(int input_size, int output_size) : weights(input_size, output_size), bias(1, output_size) {
    weights.setRandom();
    // Bias initialized to 0 by default constructor of Matrix (which fills with 0)
}

Matrix Dense::forward(const Matrix& input_mat) {
    this->input = input_mat;
    // Y = X * W + B
    Matrix output = Matrix::multiply(input, weights);
    
    // Add bias to every row
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.data[i][j] += bias.data[0][j];
        }
    }
    return output;
}

Matrix Dense::backward(const Matrix& output_gradient, double learning_rate) {
    // dE/dW = X^T * dE/dY
    Matrix weights_gradient = Matrix::multiply(input.transpose(), output_gradient);
    
    // dE/dB = sum(dE/dY)
    Matrix bias_gradient(1, bias.cols);
    for (int j = 0; j < output_gradient.cols; ++j) {
        double sum = 0;
        for (int i = 0; i < output_gradient.rows; ++i) {
            sum += output_gradient.data[i][j];
        }
        bias_gradient.data[0][j] = sum;
    }

    // dE/dX = dE/dY * W^T
    Matrix input_gradient = Matrix::multiply(output_gradient, weights.transpose());

    // Update parameters
    weights = weights - (weights_gradient * learning_rate);
    bias = bias - (bias_gradient * learning_rate);

    return input_gradient;
}
