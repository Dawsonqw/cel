#include "matrix_add.h"
#include <iostream>

__global__ void add_matrices_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

void add_matrices(const float* A, const float* B, float* C, int rows, int cols) {
    // 计算网格和块的大小
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // 定义设备上的指针
    float* d_A, *d_B, *d_C;

    // 分配设备内存
    cudaMalloc(&d_A, rows * cols * sizeof(float));
    cudaMalloc(&d_B, rows * cols * sizeof(float));
    cudaMalloc(&d_C, rows * cols * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // 启动内核
    add_matrices_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    // 检查内核执行是否成功
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}