#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Layer.h"
#include <cmath>
#include <functional>

class Activation : public Layer {
private:
    std::function<double(double)> activation;
    std::function<double(double)> activation_prime;
    Matrix input;

public:
    Activation(std::function<double(double)> act, std::function<double(double)> act_prime)
        : activation(act), activation_prime(act_prime) {}

    Matrix forward(const Matrix& input_mat) override {
        this->input = input_mat;
        Matrix output(input_mat.rows, input_mat.cols);
        for (int i = 0; i < input_mat.rows; ++i) {
            for (int j = 0; j < input_mat.cols; ++j) {
                output.data[i][j] = activation(input_mat.data[i][j]);
            }
        }
        return output;
    }

    Matrix backward(const Matrix& output_gradient, double learning_rate) override {
        Matrix input_gradient(output_gradient.rows, output_gradient.cols);
        for (int i = 0; i < output_gradient.rows; ++i) {
            for (int j = 0; j < output_gradient.cols; ++j) {
                input_gradient.data[i][j] = output_gradient.data[i][j] * activation_prime(input.data[i][j]);
            }
        }
        return input_gradient;
    }
};

class Tanh : public Activation {
public:
    Tanh() : Activation(
        [](double x) { return std::tanh(x); },
        [](double x) { double t = std::tanh(x); return 1 - t * t; }
    ) {}
};

class Sigmoid : public Activation {
public:
    Sigmoid() : Activation(
        [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
        [](double x) { double s = 1.0 / (1.0 + std::exp(-x)); return s * (1 - s); }
    ) {}
};

#endif // ACTIVATION_H
