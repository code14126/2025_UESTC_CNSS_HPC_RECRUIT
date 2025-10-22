#include "cblas.h"
#include <chrono>
#include <iostream>
#include <vector>

int main()
{
    int M = 2048, N = 1024, K = 2048;
    std::vector<float> A(M * N), B(N * K), C(M * K, 0.0f);

    // 初始化矩阵
    for (int i = 0; i < M * N; ++i)
        A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < N * K; ++i)
        B[i] = (float)((i * 2) % 100) / 100.0f;

    // 调用 BLAS
    // C = 1.0*A*B + 0.0*C
    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(
        CblasRowMajor, // 行优先存储
        CblasNoTrans,  // A 不转置
        CblasNoTrans,  // B 不转置
        M, K, N,       // 矩阵尺寸
        1.0f,          // alpha
        A.data(), N,   // A, lda
        B.data(), K,   // B, ldb
        0.0f,          // beta
        C.data(), K    // C, ldc
    );
    auto end = std::chrono::high_resolution_clock::now();

    double time_blas = std::chrono::duration<double>(end - start).count();
    std::cout << "BLAS time: " << time_blas << " s\n";

    // 验证
    std::cout << "C[0]: " << C[0] << ", C[M*K-1]: " << C[M * K - 1] << "\n";
    return 0;
}
