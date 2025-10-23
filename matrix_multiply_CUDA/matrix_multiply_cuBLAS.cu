#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>

void matmul_cublas(const float *A, const float *B, float *C, int M, int N, int K)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 默认是列主序（Fortran-style）
    // 所以我们计算 C = A * B 相当于调用 GEMM(B, A)
    // 即: C(M×K) = A(M×N) * B(N×K)
    // 需要调整顺序：C = Bᵗ * Aᵗ

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        K, M, N,              // C = (K×M) matrix
        &alpha,
        B, K,                 // B: K×N
        A, N,                 // A: N×M
        &beta,
        C, K                  // C: K×M
    );

    cublasDestroy(handle);
}

int main()
{
    int M = 2048, N = 1024, K = 2048;

    std::vector<float> A(M * N), B(N * K), C(M * K, 0.0f);

    // 初始化矩阵
    for (int i = 0; i < M * N; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 分配显存
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * N * sizeof(float));
    cudaMalloc(&dB, N * K * sizeof(float));
    cudaMalloc(&dC, M * K * sizeof(float));

    cudaMemcpy(dA, A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    matmul_cublas(dA, dB, dC, M, N, K);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "cuBLAS time: " << elapsed << " ms" << std::endl;

    cudaMemcpy(C.data(), dC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
