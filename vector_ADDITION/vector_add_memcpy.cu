#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// CUDA 错误检查
inline void checkCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }

// ==========================================
// 向量加法核函数
// ==========================================
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// ==========================================
// 使用 cudaMemcpy 模拟加法的版本
// 实际上不计算，只做数据传输
// ==========================================
void vector_copy_equivalent(const float* A, const float* B, float* C, int N) {
    // 模拟只做“传输”的操作
    // 实际场景：C <- A （或者 B）
    // 因为 cudaMemcpy 已经能达到带宽极限
    CHECK_CUDA_ERROR(cudaMemcpy(C, A, N * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ==========================================
// 主程序
// ==========================================
int main() {
    const int N = 100000000; // 1e8 元素
    const size_t bytes = N * sizeof(float);

    printf("=== 向量加法 vs cudaMemcpy 对比 ===\n");
    printf("向量长度: %d（约 %.2f MB/向量）\n", N, bytes / 1024.0 / 1024.0);

    // 分配 pinned host 内存
    float *h_A, *h_B, *h_C;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C, bytes));

    // 初始化
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes));

    // Host -> Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ===================================================
    // 1️⃣ 传统 kernel 版本计时
    // ===================================================
    cudaEvent_t start1, stop1;
    CHECK_CUDA_ERROR(cudaEventCreate(&start1));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop1));

    CHECK_CUDA_ERROR(cudaEventRecord(start1));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop1));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop1));

    float time_kernel_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_kernel_ms, start1, stop1));
    printf("kernel 向量加法时间: %.3f ms\n", time_kernel_ms);

    // ===================================================
    // 2️⃣ cudaMemcpy 版本计时
    // ===================================================
    cudaEvent_t start2, stop2;
    CHECK_CUDA_ERROR(cudaEventCreate(&start2));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop2));

    CHECK_CUDA_ERROR(cudaEventRecord(start2));
    vector_copy_equivalent(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop2));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop2));

    float time_copy_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_copy_ms, start2, stop2));
    printf("cudaMemcpy(DeviceToDevice) 时间: %.3f ms\n", time_copy_ms);

    // ===================================================
    // 3️⃣ 对比带宽
    // ===================================================
    double gb = bytes / 1e9;
    double bandwidth_kernel = gb * 3 / (time_kernel_ms / 1e3); // A+B->C 三次访问
    double bandwidth_copy = gb * 2 / (time_copy_ms / 1e3);     // A->C 两次访问

    printf("kernel 近似带宽: %.2f GB/s\n", bandwidth_kernel);
    printf("cudaMemcpy 实测带宽: %.2f GB/s\n", bandwidth_copy);

    // 清理
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start1));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop1));
    CHECK_CUDA_ERROR(cudaEventDestroy(start2));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop2));

    return 0;
}
