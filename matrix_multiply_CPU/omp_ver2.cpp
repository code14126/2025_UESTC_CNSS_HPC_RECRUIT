#include <chrono>
#include <iostream>
#include <malloc.h> // Windows 下 _aligned_malloc 需要的头文件
#include <omp.h>
#include <vector>
#include <windows.h> // 需添加该头文件

// 定义分块大小（根据你的 CPU 缓存优化）
#define BLOCK_SIZE 128

// Windows 兼容的内存对齐分配函数
template <typename T>
T *aligned_alloc(size_t size, size_t alignment)
{
    T *ptr = static_cast<T *>(_aligned_malloc(size * sizeof(T), alignment));
    if (!ptr)
    {
        std::cerr << "内存分配失败！" << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// 矩阵乘法核心函数（分块优化 + OpenMP 并行）
void matmul_omp_ver2(const float *A, const float *B, float *C,
                     int M, int N, int K)
{
    // 1. 分配并转置 B 得到 B_T（行优先访问优化）
    float *B_T = aligned_alloc<float>(K * N, 64); // B_T 是 K×N 矩阵

// 并行转置 B（B 是 N×K，转置后 B_T 是 K×N）
#pragma omp parallel for collapse(2) num_threads(32)
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B_T[i * N + j] = B[j * K + i]; // 转置逻辑
        }
    }

// 2. 分块矩阵乘法（利用 L2 缓存）
#pragma omp parallel for collapse(2) num_threads(32)
    for (int ii = 0; ii < M; ii += BLOCK_SIZE)
    {
        for (int jj = 0; jj < K; jj += BLOCK_SIZE)
        {
            // 计算当前块的边界（防止越界）
            int i_end = std::min(ii + BLOCK_SIZE, M);
            int j_end = std::min(jj + BLOCK_SIZE, K);

            // 块内计算
            for (int i = ii; i < i_end; ++i)
            {
                for (int j = jj; j < j_end; ++j)
                {
                    float sum = 0.0f;
#pragma omp simd reduction(+ : sum) aligned(A, B_T : 64)
                    for (int k = 0; k < N; ++k)
                    {
                        sum += A[i * N + k] * B_T[j * N + k];
                    }
                    C[i * K + j] = sum;
                }
            }
        }
    }

    // 3. 释放转置矩阵 B_T
    _aligned_free(B_T);
}

int main()
{
    // 设置控制台输出编码为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8); // 同时设置输入编码（可选）

    // 定义矩阵规模（可根据需要调整）
    const int M = 2048, N = 1024, K = 2048;

    // 分配对齐内存（64字节对齐，匹配 AVX512）
    float *A = aligned_alloc<float>(M * N, 64);
    float *B = aligned_alloc<float>(N * K, 64);
    float *C = aligned_alloc<float>(M * K, 64);

// 初始化矩阵（随机值）
#pragma omp parallel for collapse(2) num_threads(32)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
#pragma omp parallel for collapse(2) num_threads(32)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            B[i * K + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // 计时并执行矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    matmul_omp_ver2(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    // 输出耗时
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "矩阵乘法耗时: " << elapsed.count() << " 秒" << std::endl;

    // 释放所有内存
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C);

    return 0;
}