#include <iostream>
#include "Network.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main() {
    // Problema XOR
    // Entradas: (0,0), (0,1), (1,0), (1,1)
    // Salidas: 0, 1, 1, 0

    Matrix X(4, 2);
    X.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    Matrix Y(4, 1);
    Y.data = {{0}, {1}, {1}, {0}};

    Network net;
    net.add(new Dense(2, 3));
    net.add(new Tanh());
    net.add(new Dense(3, 1));
    net.add(new Tanh()); 

    // Entrenar
    net.train(X, Y, 10000, 0.1);

    // Probar
    Matrix prediction = net.predict(X);
    std::cout << "\nPredictions:" << std::endl;
    prediction.print();

    // Calculate precision metrics
    int correct = 0;
    double total_error = 0.0;
    
    std::cout << "\n--- Precision Analysis ---" << std::endl;
    for (int i = 0; i < Y.rows; ++i) {
        double expected = Y.data[i][0];
        double predicted = prediction.data[i][0];
        double error = std::abs(expected - predicted);
        
        // Binary classification (threshold = 0.5)
        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;
        
        if (predicted_class == expected_class) {
            correct++;
        }
        
        total_error += error;
        
        std::cout << "Input (" << X.data[i][0] << "," << X.data[i][1] << "): "
                  << "Expected=" << expected << ", Predicted=" << predicted
                  << ", Error=" << error << std::endl;
    }
    
    // Calculate final precision
    double binary_precision = (double)correct / Y.rows * 100.0;
    double mae = total_error / Y.rows;
    double numerical_precision = (1.0 - mae) * 100.0;
    
    std::cout << "\n=== FINAL PRECISION ===" << std::endl;
    std::cout << "Binary Classification Accuracy: " << binary_precision << "% (" 
              << correct << "/" << Y.rows << " correct)" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "Numerical Precision: " << numerical_precision << "%" << std::endl;
    std::cout << "========================" << std::endl;

    return 0;
}
