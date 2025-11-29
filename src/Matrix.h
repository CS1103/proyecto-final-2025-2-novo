#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    void setRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = d(gen) * sqrt(2.0 / rows); // He initialization roughly
            }
        }
    }

    static Matrix multiply(const Matrix& A, const Matrix& B) {
        assert(A.cols == B.rows);
        Matrix C(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    C.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return C;
    }

    Matrix transpose() const {
        Matrix T(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T.data[j][i] = data[i][j];
            }
        }
        return T;
    }

    Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return res;
    }

    Matrix operator*(double scalar) const {
        Matrix res(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res.data[i][j] = data[i][j] * scalar;
            }
        }
        return res;
    }
    
    // Element-wise multiplication
    Matrix hadamard(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for(int i=0; i<rows; ++i){
            for(int j=0; j<cols; ++j){
                res.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return res;
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
};

#endif // MATRIX_H
