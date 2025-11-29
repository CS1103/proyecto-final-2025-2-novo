#ifndef MSE_H
#define MSE_H

#include "../Matrix.h"

class MSE {
public:
    static double loss(const Matrix& y_true, const Matrix& y_pred) {
        double sum = 0;
        int n = y_true.rows * y_true.cols;
        for (int i = 0; i < y_true.rows; ++i) {
            for (int j = 0; j < y_true.cols; ++j) {
                double diff = y_true.data[i][j] - y_pred.data[i][j];
                sum += diff * diff;
            }
        }
        return sum / n;
    }

    static Matrix prime(const Matrix& y_true, const Matrix& y_pred) {
        Matrix grad(y_true.rows, y_true.cols);
        int n = y_true.rows * y_true.cols;
        for (int i = 0; i < y_true.rows; ++i) {
            for (int j = 0; j < y_true.cols; ++j) {
                grad.data[i][j] = 2.0 * (y_pred.data[i][j] - y_true.data[i][j]) / n;
            }
        }
        return grad;
    }
};

#endif // MSE_H
