#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <mma.h>  // WMMA库

using namespace nvcuda;

#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

// Tensor Core 16x16x16块（half类型输入，float累加）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 声明WMMA片段：A和B用half类型，累加器用float
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为0
    wmma::fill_fragment(c_frag, 0.0f);

    // 当前线程块处理的C矩阵起始位置（16x16块）
    int c_row = blockIdx.y * WMMA_M;
    int c_col = blockIdx.x * WMMA_N;

    // 循环处理K维度的所有16x16块
    for (int k = 0; k < N; k += WMMA_K) {
        // 加载A的tile（从float转换为half）
        const float* a_ptr = A + c_row * N + k;  // A[c_row][k]起始地址
        wmma::load_matrix_sync(a_frag, reinterpret_cast<const half*>(a_ptr), N);

        // 加载B的tile（从float转换为half）
        const float* b_ptr = B + k * K + c_col;  // B[k][c_col]起始地址
        wmma::load_matrix_sync(b_frag, reinterpret_cast<const half*>(b_ptr), K);

        // Tensor Core计算：c_frag = a_frag（half） * b_frag（half） + c_frag（float）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果（从float写入C）
    float* c_ptr = C + c_row * K + c_col;
    wmma::store_matrix_sync(c_ptr, c_frag, K, wmma::mem_row_major);
}

// solve函数保持不变
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blocksPerGrid((K + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 threadsPerBlock(32, 8);  // WMMA推荐每个block 256线程（32x8），匹配warp分工

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main() {
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // 检查维度是否为16的倍数（WMMA要求）
    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        std::cerr << "矩阵维度必须是16的倍数！" << std::endl;
        return EXIT_FAILURE;
    }

    float *h_A, *h_B, *h_C;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, M*N*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, N*K*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C, M*K*sizeof(float)));

    // 初始化数据
    for(int i = 0; i < M*N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < N*K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M*N*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N*K*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M*K*sizeof(float)));

    // 数据传输
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice));

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    solve(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // 结果回传
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost));

    // 性能计算
    float gflops = 2.0f * M * N * K / (elapsed_ms * 1e6);
    std::cout << "耗时: " << elapsed_ms << " ms" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;

    // 释放资源
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}