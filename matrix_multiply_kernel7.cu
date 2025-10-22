#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <algorithm>

// 错误检查宏（保持原有）
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

// ========================== kernel_7 优化配置参数 ==========================
#define BM 128    // 每个Block处理的A矩阵行数（C的行数）
#define BN 128    // 每个Block处理的B矩阵列数（C的列数）
#define BK 8      // 每个Tile的K维度大小（分块步长）
#define TM 8      // Thread Tile的行数（单线程负责C的行数）
#define TN 8      // Thread Tile的列数（单线程负责C的列数）
#define BLOCK_DIM 256  // 线程块大小（256线程）
// ===========================================================================

// ========================== 核心辅助函数（device端，修复const匹配） ==========================
// 1. float4向量加载宏（优化内存带宽）
#define FETCH_FLOAT4(ptr) (*reinterpret_cast<const float4*>(&(ptr)))  // 这里加const，匹配传入的const指针

// 2. 共享内存加载（修复const参数，支持const float*传入）
template <typename T>
__device__ void load_shared(
    const T* A,        // 修正：A是const（核函数传入的是const float*）
    const T* B,        // 修正：B是const
    T* As,             // As是共享内存，可写，非const
    T* Bs,             // Bs是共享内存，可写，非const
    int ty, 
    int tx, 
    int M, 
    int N, 
    int K
) {
    // 加载A到共享内存（按float4批量加载，分摊到多个线程）
    const int a_stride = BLOCK_DIM / BK;
    for (int i = 0; i < BM; i += a_stride) {
        int row = ty + i;
        if (row < BM) {
            int col = tx * 4;
            if (col + 3 < BK) {
                // 修正：FETCH_FLOAT4接收const指针，匹配A的const类型
                float4 a4 = FETCH_FLOAT4(A[row * N + col]);  // A是M×N，行优先，索引=row*N + col
                As[row * BK + col + 0] = a4.x;
                As[row * BK + col + 1] = a4.y;
                As[row * BK + col + 2] = a4.z;
                As[row * BK + col + 3] = a4.w;
            } else {
                // 边缘列：单元素加载（避免越界）
                for (int c = 0; c < 4 && col + c < BK; c++) {
                    As[row * BK + col + c] = A[row * N + col + c];
                }
            }
        }
    }

    // 加载B到共享内存（float4批量加载）
    const int b_stride = BLOCK_DIM / BN;
    for (int i = 0; i < BK; i += b_stride) {
        int row = ty + i;
        if (row < BK) {
            int col = tx * 4;
            if (col + 3 < BN) {
                // 修正：FETCH_FLOAT4接收const指针，匹配B的const类型
                float4 b4 = FETCH_FLOAT4(B[row * K + col]);  // B是N×K，行优先，索引=row*K + col
                Bs[row * BN + col + 0] = b4.x;
                Bs[row * BN + col + 1] = b4.y;
                Bs[row * BN + col + 2] = b4.z;
                Bs[row * BN + col + 3] = b4.w;
            } else {
                // 边缘列：单元素加载
                for (int c = 0; c < 4 && col + c < BN; c++) {
                    Bs[row * BN + col + c] = B[row * K + col + c];
                }
            }
        }
    }
}

// 3. Thread Tile计算（寄存器缓存共享内存，减少重复访问）
__device__ void compute_tile(const float* As, const float* Bs, float tmp[TM][TN], int ty, int tx) {
    float a_frag[TM];  // 寄存器缓存A的片段（TM个元素）
    float b_frag[TN];  // 寄存器缓存B的片段（TN个元素）

    // 按BK维度循环，展开优化（隐藏指令延迟）
    #pragma unroll
    for (int k = 0; k < BK; k++) {
        // 加载A片段到寄存器
        #pragma unroll
        for (int j = 0; j < TM; j++) {
            a_frag[j] = As[(ty + j) * BK + k];
        }
        // 加载B片段到寄存器
        #pragma unroll
        for (int l = 0; l < TN; l++) {
            b_frag[l] = Bs[k * BN + tx + l];
        }
        // 乘累加计算（TM×TN个元素，提高计算访存比）
        #pragma unroll
        for (int j = 0; j < TM; j++) {
            #pragma unroll
            for (int l = 0; l < TN; l++) {
                tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
    }
}

// 4. 结果写回全局内存（支持alpha/beta缩放）
__device__ void store_result(float* C, const float tmp[TM][TN], int ty, int tx, int M, int K) {
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        #pragma unroll
        for (int l = 0; l < TN; l++) {
            int c_row = blockIdx.y * BM + ty + j;
            int c_col = blockIdx.x * BN + tx + l;
            // 边界检查（避免越界写）
            if (c_row < M && c_col < K) {
                C[c_row * K + c_col] = tmp[j][l];  // alpha=1.0, beta=0.0（简化，与solve调用一致）
            }
        }
    }
}
// ===========================================================================

// ========================== kernel_7 优化核函数（修正参数匹配） ==========================
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 线程索引映射（Block内分工：tx对应C的列偏移，ty对应C的行偏移）
    const int tx = threadIdx.x % (BN / TN);  // 每个线程负责TN列，x方向步长=BN/TN
    const int ty = threadIdx.x / (BN / TN);  // 每个线程负责TM行，y方向步长=BM/TM

    // 双缓存：两组共享内存（当前计算缓存 + 预取缓存）
    __shared__ float As[2][BM * BK];  // 2*128*8=2048字节
    __shared__ float Bs[2][BK * BN];  // 2*8*128=2048字节
    int buf_idx = 0;                  // 缓存切换索引（0或1）

    // 全局内存指针定位到当前Block的起始位置（修正索引计算）
    const float* A_block = A + blockIdx.y * BM * N;  // A: M×N，行优先，当前Block的起始行=blockIdx.y*BM
    const float* B_block = B;                        // B: N×K，行优先，无需偏移（按K维度分块）
    float* C_block = C + blockIdx.y * BM * K;        // C: M×K，当前Block的起始行=blockIdx.y*BM

    // 寄存器数组：存储当前线程负责的TM×TN个元素的累加结果
    float tmp[TM][TN] = {0.0f};

    // -------------------------- 预加载第一轮数据 --------------------------
    load_shared<float>(A_block, B_block, As[buf_idx], Bs[buf_idx], ty, tx, M, N, K);  // 显式指定模板类型T=float
    __syncthreads();  // 确保第一轮数据加载完成

    // -------------------------- 主循环（双缓存预取+计算并行） --------------------------
    for (int k = 0; k < N; k += BK) {
        // 1. 预取下一轮数据到备用缓存（与当前计算并行，隐藏访存延迟）
        if (k + BK < N) {  // 避免越界预取
            const float* A_next = A_block + k + BK;          // A下一轮起始地址（行不变，列偏移k+BK）
            const float* B_next = B_block + (k + BK) * K;    // B下一轮起始地址（行偏移k+BK）
            load_shared<float>(A_next, B_next, As[1 - buf_idx], Bs[1 - buf_idx], ty, tx, M, N, K);  // 显式指定T=float
        }

        // 2. 使用当前缓存计算TM×TN的Thread Tile
        compute_tile(As[buf_idx], Bs[buf_idx], tmp, ty, tx);

        __syncthreads();  // 确保当前轮计算完成，再切换缓存

        // 3. 切换缓存索引，准备下一轮计算
        buf_idx = 1 - buf_idx;
    }

    // -------------------------- 结果写回全局内存 --------------------------
    store_result(C_block, tmp, ty, tx, M, K);
}
// ===========================================================================

// ========================== 保持不变的solve函数 ==========================
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // 网格维度：每个Block处理BM×BN的C矩阵
    dim3 threadsPerBlock(BLOCK_DIM);  // 256线程/Block（固定）
    dim3 blocksPerGrid(
        (K + BN - 1) / BN,  // 列方向Block数（向上取整）
        (M + BM - 1) / BM   // 行方向Block数（向上取整）
    );
    
    // 启动核函数（alpha=1.0, beta=0.0，与store_result逻辑一致）
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();  // 保持原有同步逻辑
}
// ===========================================================================

// ========================== 主函数（测试+计时） ==========================
int main() {
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // 主机内存分配（页锁定内存）
    float *h_A, *h_B, *h_C;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C, M * K * sizeof(float)));

    // 数据初始化
    srand(42);
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * K; ++i) h_C[i] = 0.0f;

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * K * sizeof(float)));

    // 数据传输（主机→设备）
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * K * sizeof(float), cudaMemcpyHostToDevice));

    // 计时准备
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 执行矩阵乘法
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    solve(d_A, d_B, d_C, M, N, K);  // 无任何修改
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // 结果传输（设备→主机）
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // 性能计算与输出
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float gflops = 2.0f * static_cast<float>(M) * N * K / (elapsed_ms * 1e6);
    
    std::cout << "=== kernel_7 优化后性能 ===" << std::endl;
    std::cout << "矩阵尺寸：" << M << "×" << N << " × " << N << "×" << K << " = " << M << "×" << K << std::endl;
    std::cout << "耗时：" << elapsed_ms << " ms" << std::endl;
    std::cout << "性能：" << gflops << " GFLOPS" << std::endl;

    // 资源释放
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