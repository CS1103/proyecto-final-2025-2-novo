#include <iostream>
#include "Network.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main() {
    // XOR Problem
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    // Outputs: 0, 1, 1, 0

    Matrix X(4, 2);
    X.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    Matrix Y(4, 1);
    Y.data = {{0}, {1}, {1}, {0}};

    Network net;
    net.add(new Dense(2, 3));
    net.add(new Tanh());
    net.add(new Dense(3, 1));
    net.add(new Tanh()); 

    // Train
    net.train(X, Y, 10000, 0.1);

    // Test
    Matrix prediction = net.predict(X);
    std::cout << "Predictions:" << std::endl;
    prediction.print();

    return 0;
}
