#include "Network.h"
#include "losses/MSE.h"
#include <iostream>

Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void Network::add(Layer* layer) {
    layers.push_back(layer);
}

Matrix Network::predict(const Matrix& input) {
    Matrix output = input;
    for (Layer* layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Network::train(const Matrix& x_train, const Matrix& y_train, int epochs, double learning_rate) {
    for (int e = 0; e < epochs; ++e) {
        double total_error = 0;
        
        // Stochastic Gradient Descent (one sample at a time) or Batch?
        // My Matrix implementation supports batch if rows > 1.
        // Let's assume x_train is (samples, features) and y_train is (samples, output_dim).
        
        // Forward
        Matrix output = predict(x_train);
        
        // Error
        total_error = MSE::loss(y_train, output);
        
        // Backward
        Matrix grad = MSE::prime(y_train, output);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad, learning_rate);
        }

        if ((e + 1) % 100 == 0) {
            std::cout << "Epoch " << (e + 1) << "/" << epochs << " error=" << total_error << std::endl;
        }
    }
}
