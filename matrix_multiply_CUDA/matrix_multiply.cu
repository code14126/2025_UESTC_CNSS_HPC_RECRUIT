#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }

inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", 
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float sharedA[16][16];
  //__shared__ float sharedB[16][16];
    __shared__ float sharedB[16][17];  // 多1列填充，解决银行冲突

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + 15) / 16; ++t)
    {
        if (row < M && t * 16 + threadIdx.x < N)
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && t * 16 + threadIdx.y < N)
            sharedB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * K + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < 16; ++i)
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void cpu_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];
            C[i * K + j] = sum;
        }
    }
}

int main()
{
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // 分配CPU内存
    //float *h_A = new float[M * N];
    //float *h_B = new float[N * K];
    //float *h_C_gpu = new float[M * K];
    //float *h_C_cpu = new float[M * K];

    //替换CPU内存分配为页锁定内存分配
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    cudaMallocHost(&h_A, M * N * sizeof(float));  // 页锁定内存
    cudaMallocHost(&h_B, N * K * sizeof(float));
    cudaMallocHost(&h_C_gpu, M * K * sizeof(float));
    cudaMallocHost(&h_C_cpu, M * K * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < M * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // 时间测量：创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);// 记录开始时间（包含数据传输和计算）
    
    // 数据从CPU传到GPU
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 执行GPU矩阵乘法
    solve(d_A, d_B, d_C, M, N, K);
    
    // 结果从GPU传到CPU
    cudaMemcpy(h_C_gpu, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // 等待事件完成

    // 计算耗时（毫秒）
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);


    // 启用CPU计算和结果对比（调试用）
    // 大矩阵场景：注释CPU验证
    // cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, N, K);
    // float max_error = 0.0f;
    // for (int i = 0; i < M * K; ++i)
    //     max_error = std::max(max_error, std::abs(h_C_gpu[i] - h_C_cpu[i]));
    // std::cout << "最大误差: " << max_error << std::endl;

    // 输出时间
    std::cout << "总耗时: " << total_time_ms << " 毫秒" << std::endl;
    std::cout << "总耗时: " << total_time_ms / 1000.0f << " 秒" << std::endl;

    // 释放内存
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_gpu);
    cudaFreeHost(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);  // 销毁事件
    cudaEventDestroy(stop);

    return 0;
}