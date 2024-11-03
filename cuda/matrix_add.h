#ifndef MATRIX_ADD_H
#define MATRIX_ADD_H

#include <cuda_runtime.h>

// 声明矩阵加法函数
void add_matrices(const float* A, const float* B, float* C, int rows, int cols);

#endif // MATRIX_ADD_H