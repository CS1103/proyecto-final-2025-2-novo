#include "../src/Matrix.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_matrix_creation() {
    Matrix m(3, 2);
    assert(m.rows == 3);
    assert(m.cols == 2);
    std::cout << "[PASS] Matrix creation test" << std::endl;
}

void test_matrix_multiplication() {
    Matrix A(2, 3);
    A.data = {{1, 2, 3}, {4, 5, 6}};
    
    Matrix B(3, 2);
    B.data = {{7, 8}, {9, 10}, {11, 12}};
    
    Matrix C = Matrix::multiply(A, B);
    
    // Expected: [[58, 64], [139, 154]]
    assert(C.rows == 2);
    assert(C.cols == 2);
    assert(C.data[0][0] == 58);
    assert(C.data[0][1] == 64);
    assert(C.data[1][0] == 139);
    assert(C.data[1][1] == 154);
    
    std::cout << "[PASS] Matrix multiplication test" << std::endl;
}

void test_matrix_transpose() {
    Matrix A(2, 3);
    A.data = {{1, 2, 3}, {4, 5, 6}};
    
    Matrix T = A.transpose();
    
    assert(T.rows == 3);
    assert(T.cols == 2);
    assert(T.data[0][0] == 1);
    assert(T.data[0][1] == 4);
    assert(T.data[1][0] == 2);
    assert(T.data[2][0] == 3);
    
    std::cout << "[PASS] Matrix transpose test" << std::endl;
}

void test_matrix_addition() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    
    Matrix B(2, 2);
    B.data = {{5, 6}, {7, 8}};
    
    Matrix C = A + B;
    
    assert(C.data[0][0] == 6);
    assert(C.data[0][1] == 8);
    assert(C.data[1][0] == 10);
    assert(C.data[1][1] == 12);
    
    std::cout << "[PASS] Matrix addition test" << std::endl;
}

void test_matrix_subtraction() {
    Matrix A(2, 2);
    A.data = {{5, 6}, {7, 8}};
    
    Matrix B(2, 2);
    B.data = {{1, 2}, {3, 4}};
    
    Matrix C = A - B;
    
    assert(C.data[0][0] == 4);
    assert(C.data[0][1] == 4);
    assert(C.data[1][0] == 4);
    assert(C.data[1][1] == 4);
    
    std::cout << "[PASS] Matrix subtraction test" << std::endl;
}

void test_matrix_scalar_multiplication() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    
    Matrix B = A * 2.0;
    
    assert(B.data[0][0] == 2);
    assert(B.data[0][1] == 4);
    assert(B.data[1][0] == 6);
    assert(B.data[1][1] == 8);
    
    std::cout << "[PASS] Matrix scalar multiplication test" << std::endl;
}

void test_matrix_hadamard() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    
    Matrix B(2, 2);
    B.data = {{2, 3}, {4, 5}};
    
    Matrix C = A.hadamard(B);
    
    assert(C.data[0][0] == 2);
    assert(C.data[0][1] == 6);
    assert(C.data[1][0] == 12);
    assert(C.data[1][1] == 20);
    
    std::cout << "[PASS] Matrix Hadamard product test" << std::endl;
}

int main() {
    std::cout << "Running Matrix tests..." << std::endl;
    
    test_matrix_creation();
    test_matrix_multiplication();
    test_matrix_transpose();
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_scalar_multiplication();
    test_matrix_hadamard();
    
    std::cout << "\nAll Matrix tests passed!" << std::endl;
    return 0;
}
