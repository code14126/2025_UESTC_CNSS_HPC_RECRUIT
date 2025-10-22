#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

void matmul_single(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];
            C[i * K + j] = sum;
        }
}

int main()
{
    int M = 2048, N = 1024, K = 2048;
    std::vector<float> A(M * N), B(N * K), C(M * K);

    // 初始化矩阵
    for (int i = 0; i < M * N; ++i)
        A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < N * K; ++i)
        B[i] = (float)((i * 2) % 100) / 100.0f;

    auto start = std::chrono::high_resolution_clock::now();
    matmul_single(A.data(), B.data(), C.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    double time_single = std::chrono::duration<double>(end - start).count();
    std::cout << "Single-thread time: " << time_single << " s\n";

    return 0;
}
