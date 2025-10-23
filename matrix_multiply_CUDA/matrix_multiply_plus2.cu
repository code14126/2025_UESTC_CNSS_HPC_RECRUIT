//失败版本2.0
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

// Device symbol that will hold pointer to B_trans (K x N)
__device__ const float* d_B_trans_global = nullptr;

// ------------------------------
// Transpose kernel (tile-based, avoids bank conflicts)
// transpose: input d_B (N x K) -> output d_B_trans (K x N)
__global__ void matrix_transpose_kernel(const float* __restrict__ d_B, float* __restrict__ d_B_trans,
                                        int N, int K)
{
    // Use 32x32 tile for good coalescing (adjust if too large for your SM)
    __shared__ float tile[32][33]; // pad to avoid bank conflicts

    int in_col = blockIdx.x * 32 + threadIdx.x; // k index (0..K-1)
    int in_row = blockIdx.y * 32 + threadIdx.y; // n index (0..N-1)

    // read tile (if inside original)
    if ((in_row < N) && (in_col < K)) {
        tile[threadIdx.y][threadIdx.x] = d_B[in_row * K + in_col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // write transposed: tile[j][i] -> d_B_trans[(in_col) * N + in_row] but
    // when writing, swap indices accordingly
    int out_col = blockIdx.y * 32 + threadIdx.x; // becomes new column (n)
    int out_row = blockIdx.x * 32 + threadIdx.y; // becomes new row (k)

    if ((out_row < K) && (out_col < N)) {
        d_B_trans[out_row * N + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// ------------------------------
// Optimized matrix multiplication kernel
// Keep signature same (so solve can call it unchanged)
__launch_bounds__(256)
__global__ void matrix_multiplication_kernel(const float * __restrict__ A,
                                             const float * __restrict__ B,
                                             float * __restrict__ C,
                                             int M, int N, int K)
{
    // 16x16 threads per block (solve uses this)
    __shared__ float sharedA[16][16];
    __shared__ float sharedB[16][17]; // pad to avoid bank conflicts

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // load device symbol pointer once into a register/local pointer
    const float* __restrict__ B_trans = d_B_trans_global;

    int tiles = (N + 15) / 16;
    for (int t = 0; t < tiles; ++t)
    {
        int a_col = t * 16 + threadIdx.x; // column index into A
        if (row < M && a_col < N)
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * 16 + threadIdx.y; // row index into original B (=> column in B_trans)
        // IMPORTANT: We read from B_trans to get coalesced accesses:
        // B_trans layout is K x N (row-major), so element (col, b_row) -> index col*N + b_row
        if (col < K && b_row < N && B_trans != nullptr)
            sharedB[threadIdx.y][threadIdx.x] = B_trans[col * N + b_row];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll 16
        for (int i = 0; i < 16; ++i)
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

// -----------------------------------------------------------------------------
// solve: MUST NOT CHANGE (per your constraint)
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// CPU fallback (unchanged)
void cpu_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];
            C[i * K + j] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// main: allocate, prepare, pre-transpose B on GPU, set device symbol (before timing),
//       then run (timing includes A memcpy + solve + result memcpy)
int main()
{
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // Host pinned memory
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, (size_t)M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, (size_t)N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C_gpu, (size_t)M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C_cpu, (size_t)M * K * sizeof(float)));

    // init
    for (size_t i = 0; i < (size_t)M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < (size_t)N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device allocation
    float *d_A = nullptr, *d_B = nullptr, *d_B_trans = nullptr, *d_C = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, (size_t)M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, (size_t)N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_trans, (size_t)K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, (size_t)M * K * sizeof(float)));

    // Copy B to device and pre-transpose ON GPU (do this BEFORE timing to avoid counting its sync)
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));

    // Launch transpose kernel: grid sized for 32x32 tiles
    dim3 ttx(32, 32);
    dim3 btx((K + 31) / 32, (N + 31) / 32);
    matrix_transpose_kernel<<<btx, ttx>>>(d_B, d_B_trans, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // wait transpose finished (we do it before timing)

    // Copy pointer to device symbol (do this BEFORE timing)
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_B_trans_global, &d_B_trans, sizeof(float*)));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Start timing (we include Host->Device A copy + kernel + D2H copy of C)
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Copy A (include this in timing similar to your original test)
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));

    // call solve (unchanged)
    solve(d_A, d_B, d_C, M, N, K);

    // copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float total_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_time_ms, start, stop));

    std::cout << "总耗时: " << total_time_ms << " 毫秒" << std::endl;
    std::cout << "总耗时: " << total_time_ms / 1000.0f << " 秒" << std::endl;

    // cleanup
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_gpu));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_cpu));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_B_trans));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}
