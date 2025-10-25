#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random>

// 错误检查宏（关键！大规模运算易出现内存/配置错误）
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA错误：" << cudaGetErrorString(err) << "（行号：" << __LINE__ << "）" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
    CHECK(cudaGetLastError()); // 检查核函数启动错误
    cudaDeviceSynchronize();
}

int main() {
    // 向量长度：1亿（100,000,000）
    const int N = 100000000;
    std::cout << "向量长度: " << N << "（约" << (N * sizeof(float) / 1024 / 1024) << "MB/向量）" << std::endl;

    // 主机端内存分配（动态分配，避免栈溢出）
    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    if (!h_A || !h_B || !h_C) {
        std::cerr << "主机内存分配失败！" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --------------------------
    // 1. 生成随机数据（计时）
    // --------------------------
    auto start_gen = std::chrono::high_resolution_clock::now();
    
    // 高效随机数生成（比rand()快，适合大规模数据）
    std::mt19937 gen(time(0)); // 随机数引擎
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f); // 0~1000的随机浮点数
    for (int i = 0; i < N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
    
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_time = end_gen - start_gen;
    std::cout << "数据生成时间: " << gen_time.count() << " s" << std::endl;

    // --------------------------
    // 2. 设备内存分配
    // --------------------------
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // --------------------------
    // 3. 主机->设备数据传输（计时）
    // --------------------------
    cudaEvent_t start_h2d, end_h2d;
    CHECK(cudaEventCreate(&start_h2d));
    CHECK(cudaEventCreate(&end_h2d));
    CHECK(cudaEventRecord(start_h2d, 0));

    CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(end_h2d, 0));
    CHECK(cudaEventSynchronize(end_h2d));
    float h2d_time_ms;
    CHECK(cudaEventElapsedTime(&h2d_time_ms, start_h2d, end_h2d));
    std::cout << "主机到设备传输时间: " << h2d_time_ms << " ms" << std::endl;

    // --------------------------
    // 4. GPU计算（计时）
    // --------------------------
    cudaEvent_t start_kernel, end_kernel;
    CHECK(cudaEventCreate(&start_kernel));
    CHECK(cudaEventCreate(&end_kernel));
    CHECK(cudaEventRecord(start_kernel, 0));

    solve(d_A, d_B, d_C, N); // 调用GPU计算

    CHECK(cudaEventRecord(end_kernel, 0));
    CHECK(cudaEventSynchronize(end_kernel));
    float kernel_time_ms;
    CHECK(cudaEventElapsedTime(&kernel_time_ms, start_kernel, end_kernel));
    std::cout << "GPU计算时间: " << kernel_time_ms << " ms" << std::endl;

    // --------------------------
    // 5. 设备->主机数据传输（计时）
    // --------------------------
    cudaEvent_t start_d2h, end_d2h;
    CHECK(cudaEventCreate(&start_d2h));
    CHECK(cudaEventCreate(&end_d2h));
    CHECK(cudaEventRecord(start_d2h, 0));

    CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(end_d2h, 0));
    CHECK(cudaEventSynchronize(end_d2h));
    float d2h_time_ms;
    CHECK(cudaEventElapsedTime(&d2h_time_ms, start_d2h, end_d2h));
    std::cout << "设备到主机传输时间: " << d2h_time_ms << " ms" << std::endl;

    // --------------------------
    // 6. 抽样验证结果（大规模数据无需全量检查）
    // --------------------------
    const int check_count = 100; // 随机抽查100个元素
    const float eps = 1e-5f; // 浮点误差阈值
    bool success = true;
    std::cout << "随机抽查" << check_count << "个元素验证..." << std::endl;

    for (int i = 0; i < check_count; ++i) {
        int idx = rand() % N; // 随机索引
        float expected = h_A[idx] + h_B[idx];
        if (fabs(h_C[idx] - expected) > eps) {
            success = false;
            std::cerr << "验证失败：索引 " << idx 
                      << "，计算值 " << h_C[idx] 
                      << "，期望值 " << expected << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "所有抽查元素验证通过，结果正确。" << std::endl;
    }

    // --------------------------
    // 7. 释放资源
    // --------------------------
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaEventDestroy(start_h2d));
    CHECK(cudaEventDestroy(end_h2d));
    CHECK(cudaEventDestroy(start_kernel));
    CHECK(cudaEventDestroy(end_kernel));
    CHECK(cudaEventDestroy(start_d2h));
    CHECK(cudaEventDestroy(end_d2h));

    return 0;
}