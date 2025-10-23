// matrix_multiply_kernel7_fixed.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>

// 错误检查宏
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

// ========================== kernel_7 优化配置参数 ==========================
#define BM 128    // Block 负责 C 的行数
#define BN 128    // Block 负责 C 的列数
#define BK 8      // tile 在 K 方向的大小
#define TM 8      // 单线程负责的行数
#define TN 8      // 单线程负责的列数
#define BLOCK_DIM 256  // 线程数/Block（保持 256）
// =======================================================================

// device 辅助：按 tile 将 A_block, B 加载进共享内存（按线程 stride），并做 bounds check
__device__ void load_shared_tile(
    const float* A_block_base, // 指向 A 的当前 block 行起始（即 A + block_row*BM*N）
    const float* B,           // 指向整个 B（N x K）
    float* As,                // 共享内存 As (BM x BK)
    float* Bs,                // 共享内存 Bs (BK x BN)
    int k_tile,               // 当前 tile 在 K 方向起始（0, BK, 2*BK, ...）
    int block_row,            // blockIdx.y
    int block_col,            // blockIdx.x
    int M, int N, int K
) {
    int tid = threadIdx.x;

    // As: BM rows * BK cols
    const int As_elems = BM * BK;
    for (int idx = tid; idx < As_elems; idx += BLOCK_DIM) {
        int local_row = idx / BK;    // 0..BM-1
        int local_col = idx % BK;    // 0..BK-1
        int global_row = block_row * BM + local_row; // A 全局行
        int global_col = k_tile + local_col;         // A 全局列 (in [0, N))
        if (global_row < M && global_col < N) {
            // A_block_base 指向 A + block_row*BM*N
            As[local_row * BK + local_col] = A_block_base[ local_row * N + local_col + k_tile ];
        } else {
            As[local_row * BK + local_col] = 0.0f;
        }
    }

    // Bs: BK rows * BN cols  (B 的行是 K-dimension tile over N)
    const int Bs_elems = BK * BN;
    for (int idx = tid; idx < Bs_elems; idx += BLOCK_DIM) {
        int local_row = idx / BN;   // 0..BK-1
        int local_col = idx % BN;   // 0..BN-1
        int global_row = k_tile + local_row;              // B 全局行 (in [0, N))
        int global_col = block_col * BN + local_col;      // B 全局列 (in [0, K))
        if (global_row < N && global_col < K) {
            Bs[local_row * BN + local_col] = B[ global_row * K + global_col ];
        } else {
            Bs[local_row * BN + local_col] = 0.0f;
        }
    }
}

// Thread-tile 计算：As (BM x BK), Bs (BK x BN) -> tmp (TM x TN)
// thread_tile_row/col 表示该线程负责的子块在共享内存里的起始位置（以元素计）
__device__ void compute_tile(const float* As, const float* Bs, float tmp[TM][TN],
                             int thread_tile_row, int thread_tile_col) {
    // tmp 已由调用方初始化为 0
    // 遍历 BK
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        float a_frag[TM];
        float b_frag[TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            // As layout: BM x BK  -> index = (thread_tile_row + i) * BK + k
            a_frag[i] = As[(thread_tile_row + i) * BK + k];
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            // Bs layout: BK x BN -> index = k * BN + (thread_tile_col + j)
            b_frag[j] = Bs[k * BN + (thread_tile_col + j)];
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                tmp[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }
}

// 结果写回 C（带边界检查）
__device__ void store_result(float* C, const float tmp[TM][TN],
                             int block_row, int block_col, int thread_tile_row, int thread_tile_col,
                             int M, int K) {
    // 每个线程负责 TM x TN 的输出
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int c_row = block_row * BM + thread_tile_row + i;
            int c_col = block_col * BN + thread_tile_col + j;
            if (c_row < M && c_col < K) {
                C[c_row * K + c_col] = tmp[i][j];
            }
        }
    }
}

// Kernel：每个 block 处理 BM x BN 的 C 子矩阵
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 线程组织：BLOCK_DIM = 256
    // 我们把线程分成 16 x 16 的逻辑网格 (因为 BN/TN = 16, BM/TM = 16)
    const int threads_x = BN / TN; // 128/8 = 16
    // tx in [0..threads_x-1], ty in [0..threads_x-1]
    int local_tid = threadIdx.x;
    int tx = local_tid % threads_x;
    int ty = local_tid / threads_x;

    // 每线程负责的起始位置（以元素计）
    int thread_tile_row = ty * TM;    // in [0, BM)
    int thread_tile_col = tx * TN;    // in [0, BN)

    // 共享内存：As (BM * BK), Bs (BK * BN)
    extern __shared__ char shmem_raw[]; // 使用动态共享内存分配以避免编译器警告
    float* As = reinterpret_cast<float*>(shmem_raw); // size BM*BK
    float* Bs = As + (BM * BK);                      // size BK*BN
    // total shared size = (BM*BK + BK*BN) * sizeof(float)

    // 指向 A 的当前 block 行起始：A + blockIdx.y * BM * N
    const float* A_block_base = A + (size_t)blockIdx.y * BM * (size_t)N;
    // B 全局指针直接使用 B

    // 寄存器 tmp 存储线程负责的 TM x TN 子块
    float tmp[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            tmp[i][j] = 0.0f;

    // 遍历 K 方向的 tile
    for (int k_tile = 0; k_tile < N; k_tile += BK) {
        // 加载当前 tile 到共享内存（由所有线程分担）
        load_shared_tile(A_block_base, B, As, Bs, k_tile, blockIdx.y, blockIdx.x, M, N, K);
        __syncthreads(); // 确保共享内存加载完成

        // 计算基于当前 tile 的乘加
        compute_tile(As, Bs, tmp, thread_tile_row, thread_tile_col);
        __syncthreads(); // 确保所有线程在使用 As/Bs 完成后再改写共享内存（下轮加载）
    }

    // 写回 C
    float* C_block_base = C + (size_t)blockIdx.y * BM * (size_t)K;
    // store_result 需要 C 的全局起始地址；我们直接传递 C（并用 block_row/block_col 计算位置）
    store_result(C, tmp, blockIdx.y, blockIdx.x, thread_tile_row, thread_tile_col, M, K);
}

// solve 保持不变（但我们在 kernel 调用后增加 cudaGetLastError 检查）
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid(
        (K + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // 计算所需的动态共享内存大小
    size_t shared_bytes = (size_t)(BM * BK + BK * BN) * sizeof(float);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(A, B, C, M, N, K);

    // 捕获 launch 错误
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// ========================== 主函数（测试+计时） ==========================
int main() {
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // 分配页锁定主机内存
    float *h_A, *h_B, *h_C;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, (size_t)M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, (size_t)N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C, (size_t)M * K * sizeof(float)));

    // 初始化
    srand(42);
    for (size_t i = 0; i < (size_t)M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < (size_t)N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < (size_t)M * K; ++i) h_C[i] = 0.0f;

    // 设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, (size_t)M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, (size_t)N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, (size_t)M * K * sizeof(float)));

    // 复制
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    // 事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    solve(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // 取回结果
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double gflops = 2.0 * (double)M * (double)N * (double)K / (elapsed_ms * 1e6);

    std::cout << "=== kernel_7_fixed 性能 ===" << std::endl;
    std::cout << "矩阵尺寸：" << M << "×" << N << " × " << N << "×" << K << " = " << M << "×" << K << std::endl;
    std::cout << "耗时：" << elapsed_ms << " ms" << std::endl;
    std::cout << "性能：" << gflops << " GFLOPS" << std::endl;

    // 释放
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
