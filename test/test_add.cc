#include <iostream>
#include "matrix_add.h"

int main() {
    int rows = 4;
    int cols = 4;

    // 初始化矩阵 A 和 B
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float B[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float C[16];

    // 调用矩阵加法函数
    add_matrices(A, B, C, rows, cols);

    // 打印结果
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << C[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}