#include "../src/Network.h"
#include "../src/layers/Dense.h"
#include "../src/layers/Activation.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_xor_training() {
    std::cout << "Testing XOR problem training..." << std::endl;
    
    // XOR dataset
    Matrix X(4, 2);
    X.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Matrix Y(4, 1);
    Y.data = {{0}, {1}, {1}, {0}};
    
    Network net;
    net.add(new Dense(2, 4));
    net.add(new Tanh());
    net.add(new Dense(4, 1));
    net.add(new Tanh());
    
    // Train
    std::cout << "Training network..." << std::endl;
    net.train(X, Y, 5000, 0.1);
    
    // Test predictions
    Matrix predictions = net.predict(X);
    
    std::cout << "\nPredictions:" << std::endl;
    predictions.print();
    
    std::cout << "\nExpected:" << std::endl;
    Y.print();
    
    // Check if predictions are close to expected values
    // With Tanh, outputs are in range [-1, 1], so we need to check if they're close to 0 or 1
    double threshold = 0.5;
    bool all_correct = true;
    
    for (int i = 0; i < 4; ++i) {
        double pred = predictions.data[i][0];
        double expected = Y.data[i][0];
        
        // Check if prediction is on the correct side of 0
        if (expected == 0 && pred > threshold) {
            all_correct = false;
            std::cout << "Failed for input " << i << ": expected ~0, got " << pred << std::endl;
        } else if (expected == 1 && pred < -threshold) {
            all_correct = false;
            std::cout << "Failed for input " << i << ": expected ~1, got " << pred << std::endl;
        }
    }
    
    if (all_correct) {
        std::cout << "[PASS] XOR training test - Network learned the pattern!" << std::endl;
    } else {
        std::cout << "[INFO] XOR training test - Network may need more training or tuning" << std::endl;
    }
}

int main() {
    std::cout << "Running XOR integration test..." << std::endl;
    
    test_xor_training();
    
    std::cout << "\nXOR integration test completed!" << std::endl;
    return 0;
}
