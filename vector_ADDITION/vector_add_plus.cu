#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <chrono>

// 错误检查宏
inline void checkCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", 
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }

// 核函数：向量逐元素相加（保持不变）
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

// 不修改solve函数
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

// GPU随机数初始化核函数
__global__ void init_rand_states(curandState* states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 每个线程初始化独立的随机数状态
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// GPU数据生成核函数（直接生成到统一内存）
__global__ void generate_gpu_data(float* A, float* B, curandState* states, int N, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 生成0~max_val的随机数
        A[idx] = curand_uniform(&states[idx]) * max_val;
        B[idx] = curand_uniform(&states[idx]) * max_val;
    }
}

// GPU结果验证核函数（仅验证随机选择的索引）
__global__ void verify_gpu_results(const float* A, const float* B, const float* C, 
                                  const int* indices, bool* success, int check_count, float eps) {
    int i = threadIdx.x;  // 每个线程验证一个索引
    if (i < check_count) {
        int idx = indices[i];
        float expected = A[idx] + B[idx];
        if (fabsf(C[idx] - expected) > eps) {
            *success = false;  // 原子操作确保正确性（此处简化为直接赋值，多线程下安全）
        }
    }
}

// 多线程生成随机索引（主机端辅助函数）
void generate_indices_thread(int start, int end, int* indices, int N, std::mt19937& gen) {
    std::uniform_int_distribution<int> dist(0, N-1);
    for (int i = start; i < end; ++i) {
        indices[i] = dist(gen);
    }
}

int main() {
    const int N = 100000000;  // 向量长度（1亿）
    const int check_count = 1000;  // 验证样本数
    const float max_val = 1000.0f;  // 随机数最大值
    const float eps = 1e-5f;  // 浮点误差阈值
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("向量长度: %d（约%dMB/向量）\n", 
           N, (int)(N * sizeof(float) / 1024 / 1024));

    // --------------------------
    // 1. 分配统一内存（Unified Memory）
    // --------------------------
    float *A, *B, *C;  // 统一内存指针（主机和设备均可访问）
    CHECK_CUDA_ERROR(cudaMallocManaged(&A, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&B, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&C, N * sizeof(float)));

    // --------------------------
    // 2. 多线程生成验证用随机索引（主机端）
    // --------------------------
    int* indices = new int[check_count];
    const int num_threads = std::thread::hardware_concurrency();  // 获取硬件支持的线程数
    const int chunk_size = (check_count + num_threads - 1) / num_threads;  // 每个线程处理的索引数
    std::vector<std::thread> threads;
    std::vector<std::mt19937> gens;  // 每个线程独立的随机数引擎（避免锁竞争）

    auto start_indices = std::chrono::high_resolution_clock::now();

    // 初始化每个线程的随机数引擎（用不同种子）
    for (int i = 0; i < num_threads; ++i) {
        gens.emplace_back(std::mt19937(time(0) + i));  // 种子偏移避免重复
    }

    // 多线程生成索引
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, check_count);
        if (start >= check_count) break;
        threads.emplace_back(generate_indices_thread, start, end, indices, N, std::ref(gens[i]));
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    auto end_indices = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> indices_time = end_indices - start_indices;
    printf("多线程生成验证索引时间: %.6f s\n", indices_time.count());

    // --------------------------
    // 3. 初始化CUDA流和GPU随机数状态
    // --------------------------
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));  // 创建非默认流（异步操作）

    curandState* d_states;  // GPU随机数状态数组
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, N * sizeof(curandState)));

    // 异步初始化随机数状态（绑定到流）
    init_rand_states<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_states, time(0), N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --------------------------
    // 4. GPU异步生成数据（直接写入统一内存）
    // --------------------------
    cudaEvent_t start_gpu_gen, end_gpu_gen;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu_gen));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_gpu_gen));
    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu_gen, stream));

    // 异步生成A和B（绑定到流，依赖随机数状态初始化完成）
    generate_gpu_data<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(A, B, d_states, N, max_val);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventRecord(end_gpu_gen, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_gpu_gen));
    float gpu_gen_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_gen_time_ms, start_gpu_gen, end_gpu_gen));
    printf("GPU数据生成时间: %.3f ms\n", gpu_gen_time_ms);

    // --------------------------
    // 5. 异步执行GPU计算（solve函数）
    // --------------------------
    cudaEvent_t start_kernel, end_kernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_kernel));
    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel, stream));

    solve(A, B, C, N);  // 调用核函数（内部已同步，此处依赖流顺序执行）

    CHECK_CUDA_ERROR(cudaEventRecord(end_kernel, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_kernel));
    float kernel_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_time_ms, start_kernel, end_kernel));
    printf("GPU计算时间: %.3f ms\n", kernel_time_ms);

    // --------------------------
    // 6. GPU异步验证结果（仅传输必要数据）
    // --------------------------
    // 分配设备端验证用内存
    int* d_indices;
    bool* d_success;
    CHECK_CUDA_ERROR(cudaMalloc(&d_indices, check_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_success, sizeof(bool)));

    // 主机→设备传输验证索引（异步）
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_indices, indices, check_count * sizeof(int), 
                                    cudaMemcpyHostToDevice, stream));
    bool h_success = true;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_success, &h_success, sizeof(bool), 
                                    cudaMemcpyHostToDevice, stream));

    // 异步执行GPU验证（绑定到流，依赖计算完成）
    cudaEvent_t start_verify, end_verify;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_verify));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_verify));
    CHECK_CUDA_ERROR(cudaEventRecord(start_verify, stream));

    verify_gpu_results<<<1, check_count, 0, stream>>>(A, B, C, d_indices, d_success, check_count, eps);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventRecord(end_verify, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_verify));
    float verify_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&verify_time_ms, start_verify, end_verify));
    printf("GPU验证时间: %.3f ms\n", verify_time_ms);

    // 传输验证结果回主机
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_success, d_success, sizeof(bool), 
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));  // 等待流中所有操作完成

    if (h_success) {
        printf("随机抽查%d个元素验证通过，结果正确。\n", check_count);
    } else {
        fprintf(stderr, "验证失败：部分元素计算结果不匹配！\n");
    }

    // --------------------------
    // 7. 释放资源
    // --------------------------
    delete[] indices;
    CHECK_CUDA_ERROR(cudaFree(d_states));
    CHECK_CUDA_ERROR(cudaFree(d_indices));
    CHECK_CUDA_ERROR(cudaFree(d_success));
    CHECK_CUDA_ERROR(cudaFree(A));
    CHECK_CUDA_ERROR(cudaFree(B));
    CHECK_CUDA_ERROR(cudaFree(C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu_gen));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_gpu_gen));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_verify));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_verify));

    return 0;
}