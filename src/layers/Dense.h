#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer {
private:
    Matrix input;

public:
    Matrix weights;
    Matrix bias;
    
    Dense(int input_size, int output_size);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_gradient, double learning_rate) override;
};

#endif // DENSE_H
