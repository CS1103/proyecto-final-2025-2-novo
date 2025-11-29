#ifndef LAYER_H
#define LAYER_H

#include "../Matrix.h"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, double learning_rate) = 0;
};

#endif // LAYER_H
