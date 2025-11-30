#include <iostream>
#include <cmath>
#include <chrono>
#include "Network.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main() {
    std::cout << "=== Large-Scale Neural Network Training Demo ===" << std::endl;
    std::cout << "This demo trains on a larger synthetic dataset to demonstrate" << std::endl;
    std::cout << "performance on more realistic problems (~2-3 minutes training time)" << std::endl;
    std::cout << std::endl;

    // Create a larger synthetic dataset: sine wave classification
    // Task: Classify if sin(x1) + cos(x2) > 0
    const int num_samples = 1000;
    const int input_dim = 10;
    const int output_dim = 1;

    std::cout << "Generating synthetic dataset..." << std::endl;
    std::cout << "Samples: " << num_samples << std::endl;
    std::cout << "Input dimensions: " << input_dim << std::endl;
    std::cout << "Output dimensions: " << output_dim << std::endl;
    std::cout << std::endl;

    Matrix X(num_samples, input_dim);
    Matrix Y(num_samples, output_dim);

    // Generate synthetic data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.14159, 3.14159);

    for (int i = 0; i < num_samples; ++i) {
        double sum = 0;
        for (int j = 0; j < input_dim; ++j) {
            X.data[i][j] = dis(gen);
            // Complex function: weighted sum of sin and cos
            if (j % 2 == 0) {
                sum += std::sin(X.data[i][j]) * (j + 1) / input_dim;
            } else {
                sum += std::cos(X.data[i][j]) * (j + 1) / input_dim;
            }
        }
        // Binary classification: is sum positive?
        Y.data[i][0] = (sum > 0) ? 1.0 : 0.0;
    }

    std::cout << "Dataset generated successfully!" << std::endl;
    std::cout << std::endl;

    // Create a deeper network: 10 -> 50 -> 30 -> 10 -> 1
    std::cout << "Building neural network architecture..." << std::endl;
    std::cout << "Architecture: " << input_dim << " -> 50 -> 30 -> 10 -> " << output_dim << std::endl;
    std::cout << std::endl;

    Network net;
    
    // Layer 1: Input to first hidden layer
    net.add(new Dense(input_dim, 50));
    net.add(new Tanh());
    
    // Layer 2: First to second hidden layer
    net.add(new Dense(50, 30));
    net.add(new Tanh());
    
    // Layer 3: Second to third hidden layer
    net.add(new Dense(30, 10));
    net.add(new Tanh());
    
    // Layer 4: Third hidden to output
    net.add(new Dense(10, output_dim));
    net.add(new Sigmoid());

    std::cout << "Network built successfully!" << std::endl;
    std::cout << "Total layers: 8 (4 Dense + 4 Activation)" << std::endl;
    std::cout << std::endl;

    // Training parameters
    const int epochs = 1000;
    const double learning_rate = 0.01;

    std::cout << "Training parameters:" << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << std::endl;

    std::cout << "Starting training..." << std::endl;
    std::cout << "(This will take approximately 2-3 minutes)" << std::endl;
    std::cout << std::endl;

    // Measure training time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Train
    net.train(X, Y, epochs, learning_rate);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << std::endl;
    std::cout << "Training completed!" << std::endl;
    std::cout << "Total training time: " << duration.count() << " seconds";
    std::cout << " (" << duration.count() / 60 << "m " << duration.count() % 60 << "s)" << std::endl;
    std::cout << std::endl;

    // Test on a subset of the data
    std::cout << "Testing on first 10 samples..." << std::endl;
    std::cout << std::endl;

    Matrix X_test(10, input_dim);
    Matrix Y_test(10, output_dim);
    
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            X_test.data[i][j] = X.data[i][j];
        }
        Y_test.data[i][0] = Y.data[i][0];
    }

    Matrix predictions = net.predict(X_test);

    // Calculate precision metrics
    int correct = 0;
    double total_error = 0.0;

    std::cout << "--- Test Results (First 10 Samples) ---" << std::endl;
    for (int i = 0; i < 10; ++i) {
        double expected = Y_test.data[i][0];
        double predicted = predictions.data[i][0];
        double error = std::abs(expected - predicted);

        // Binary classification (threshold = 0.5)
        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;

        if (predicted_class == expected_class) {
            correct++;
        }

        total_error += error;

        std::cout << "Sample " << (i + 1) << ": "
                  << "Expected=" << expected << ", Predicted=" << predicted
                  << ", Error=" << error 
                  << " [" << (predicted_class == expected_class ? "✓" : "✗") << "]"
                  << std::endl;
    }

    std::cout << std::endl;

    // Calculate final precision on test set
    double binary_precision = (double)correct / 10 * 100.0;
    double mae = total_error / 10;
    double numerical_precision = (1.0 - mae) * 100.0;

    std::cout << "=== FINAL PRECISION (Test Set) ===" << std::endl;
    std::cout << "Binary Classification Accuracy: " << binary_precision << "% ("
              << correct << "/10 correct)" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "Numerical Precision: " << numerical_precision << "%" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << std::endl;

    // Full dataset evaluation
    std::cout << "Evaluating on full dataset (" << num_samples << " samples)..." << std::endl;
    Matrix full_predictions = net.predict(X);
    
    int full_correct = 0;
    double full_total_error = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        double expected = Y.data[i][0];
        double predicted = full_predictions.data[i][0];
        double error = std::abs(expected - predicted);
        
        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;
        
        if (predicted_class == expected_class) {
            full_correct++;
        }
        
        full_total_error += error;
    }
    
    double full_binary_precision = (double)full_correct / num_samples * 100.0;
    double full_mae = full_total_error / num_samples;
    double full_numerical_precision = (1.0 - full_mae) * 100.0;
    
    std::cout << std::endl;
    std::cout << "=== FINAL PRECISION (Full Dataset) ===" << std::endl;
    std::cout << "Binary Classification Accuracy: " << full_binary_precision << "% ("
              << full_correct << "/" << num_samples << " correct)" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << full_mae << std::endl;
    std::cout << "Numerical Precision: " << full_numerical_precision << "%" << std::endl;
    std::cout << "=======================================" << std::endl;

    return 0;
}
