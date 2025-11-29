#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "layers/Layer.h"

class Network {
private:
    std::vector<Layer*> layers;

public:
    ~Network();
    void add(Layer* layer);
    Matrix predict(const Matrix& input);
    void train(const Matrix& input, const Matrix& output, int epochs, double learning_rate);
};

#endif // NETWORK_H
