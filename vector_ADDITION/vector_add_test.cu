#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// 核函数：向量逐元素相加
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

// 调用核函数的入口（设备指针版本）
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize(); // 等待GPU计算完成
}

// 主机端测试代码
int main() {
    // 1. 定义测试数据（示例向量）
    const int N = 4;
    float h_A[N] = {30.0f, 31.0f, 60.0f, 70.0f};  // 主机端A
    float h_B[N] = {20.0f, 45.0f, 12.0f, 31.9f};  // 主机端B
    float h_C[N];                                 // 主机端结果C（用于接收GPU输出）

    // 2. 分配设备内存（GPU上的内存）
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // 3. 将主机数据拷贝到设备
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. 调用GPU计算（通过solve函数）
    solve(d_A, d_B, d_C, N);

    // 5. 将设备结果拷贝回主机
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证结果（允许浮点精度误差）
    const float eps = 1e-5f; // 误差阈值
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > eps) {
            success = false;
            std::cout << "错误：索引 " << i << "，计算值 " << h_C[i] << "，期望值 " << expected << std::endl;
        }
    }

    if (success) {
        std::cout << "测试通过！结果正确。" << std::endl;
        std::cout << "输出向量C：";
        for (int i = 0; i < N; ++i) {
            std::cout << h_C[i] << " ";
        }
        std::cout << std::endl;
    }

    // 7. 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}