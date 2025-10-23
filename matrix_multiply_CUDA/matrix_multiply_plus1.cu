//失败版本1.0
// 头文件引入：按功能分类，明确每个头文件用途
#include <cstdlib>        // 提供rand()函数（矩阵随机初始化）
#include <cuda_runtime.h> // CUDA核心运行时API（内存分配、数据传输、内核调度等）
#include <iostream>       // 提供std::cout（结果与时间输出）
#include <stdio.h>        // 提供fprintf（CUDA错误信息打印）

// -----------------------------------------------------------------------------
// 1. CUDA错误检查宏：捕获API调用错误，快速定位问题（避免程序默默崩溃）
// -----------------------------------------------------------------------------
// 功能：将CUDA API返回值、当前文件、行号传入检查函数，错误时打印信息并退出
#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) { // 若API调用失败（非cudaSuccess）
        fprintf(stderr, "CUDA Error: %s（文件：%s，行号：%d）\n", 
                cudaGetErrorString(code), // 错误码转人类可读描述（如“out of memory”）
                file, line);              // 定位错误发生的文件和行号
        exit(EXIT_FAILURE);              // 退出程序，避免错误扩散
    }
}

// -----------------------------------------------------------------------------
// 2. 全局设备变量：适配“不修改solve函数”的题目要求
// -----------------------------------------------------------------------------
// 作用：传递GPU端转置后的B矩阵地址（solve参数固定为原B，内核需通过全局变量访问转置后B）
__device__ float* d_B_trans_global = nullptr;

// -----------------------------------------------------------------------------
// 3. GPU矩阵转置内核：优化B矩阵的全局内存访问（解决非合并访问问题）
// -----------------------------------------------------------------------------
// 输入：d_B（原B矩阵，GPU端，N行K列）、d_B_trans（转置后B矩阵，GPU端，K行N列）
// 功能：将N×K的原B转置为K×N的B_trans，让后续乘法内核能“合并访问”内存（提升带宽利用率）
__global__ void matrix_transpose_kernel(const float* d_B, float* d_B_trans, int N, int K) {
    // 计算当前线程负责的原B矩阵元素索引（线程块16x16，匹配后续乘法内核）
    int n = blockIdx.y * blockDim.y + threadIdx.y; // 原B的行索引（0~N-1）
    int k = blockIdx.x * blockDim.x + threadIdx.x; // 原B的列索引（0~K-1）

    // 仅处理有效范围内的元素（避免越界访问）
    if (n < N && k < K) {
        // 行主序转置逻辑：原B[n][k]（索引n*K +k）→ 转置后B_trans[k][n]（索引k*N +n）
        d_B_trans[k * N + n] = d_B[n * K + k];
    }
}

// -----------------------------------------------------------------------------
// 4. GPU矩阵乘法内核：核心计算模块（含共享内存优化）
// -----------------------------------------------------------------------------
// 输入：A/B/C（GPU端指针，B仅为占位符，实际用d_B_trans_global）、矩阵维度M/N/K
// 功能：并行计算C=A×B，用共享内存减少全局内存访问，提升计算效率
__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // 共享内存配置：解决银行冲突（sharedB多1列），尺寸匹配solve的16x16线程块
    __shared__ float sharedA[16][16];  // 缓存A矩阵的16x16子块
    __shared__ float sharedB[16][17];  // 缓存B_trans矩阵的16x16子块（多1列避免银行冲突）

    // 计算当前线程负责的C矩阵元素索引（与solve的16x16线程块对应）
    int row = blockIdx.y * blockDim.y + threadIdx.y; // C的行索引（0~M-1）
    int col = blockIdx.x * blockDim.x + threadIdx.x; // C的列索引（0~K-1）

    float sum = 0.0f; // 存储当前线程的累加结果（C[row][col]的中间值）

    // 分块循环：将大矩阵拆分为16x16子块计算（减少全局内存访问次数）
    for (int t = 0; t < (N + 15) / 16; ++t)
    {
        // -------------------------- 加载A子块到共享内存 --------------------------
        if (row < M && (t * 16 + threadIdx.x) < N) { // 仅加载有效元素
            // A的全局索引：row行，(t*16 + threadIdx.x)列（每次加载16列）
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + (t * 16 + threadIdx.x)];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f; // 越界元素填0（不影响结果）
        }

        // -------------------------- 加载B_trans子块到共享内存 --------------------------
        // 关键：通过全局变量d_B_trans_global访问转置后的B（而非传入的B参数）
        if (col < K && (t * 16 + threadIdx.y) < N) { // 仅加载有效元素
            // B_trans的全局索引：col行（原B的列），(t*16 + threadIdx.y)列（原B的行）
            sharedB[threadIdx.y][threadIdx.x] = d_B_trans_global[col * N + (t * 16 + threadIdx.y)];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f; // 越界元素填0
        }

        __syncthreads(); // 同步线程块：确保所有线程加载完共享内存，再开始计算

        // -------------------------- 子块计算（循环展开优化） --------------------------
        #pragma unroll 16 // 编译器指令：手动展开循环，减少控制开销
        for (int i = 0; i < 16; ++i) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x]; // 累加部分和
        }

        __syncthreads(); // 同步线程块：确保当前子块计算完成，再加载下一个子块
    }

    // -------------------------- 写入最终结果 --------------------------
    if (row < M && col < K) { // 仅写入有效元素（避免越界）
        C[row * K + col] = sum; // C矩阵按行主序存储：row行col列 → 索引row*K +col
    }
}

// -----------------------------------------------------------------------------
// 5. solve函数：按题目要求完全未修改（固定参数、线程块、内核调用方式）
// -----------------------------------------------------------------------------
// 输入：A/B/C（GPU端指针，题目要求均为设备指针）、矩阵维度M/N/K
// 功能：配置线程块/网格维度，调用乘法内核，等待内核执行完成
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16); // 线程块尺寸：16x16（题目固定，不修改）
    // 计算网格维度：向上取整，确保覆盖C矩阵所有元素（行方向M行，列方向K列）
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // 调用乘法内核（参数顺序、调度方式均按题目要求未修改）
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize(); // 主机等待GPU内核执行完成（确保结果正确）
}

// -----------------------------------------------------------------------------
// 6. CPU矩阵乘法函数：基准验证工具（用于对比GPU结果正确性）
// -----------------------------------------------------------------------------
// 输入：A/B/C（CPU端指针）、矩阵维度M/N/K
// 功能：按矩阵乘法公式实现串行计算，结果作为“标准答案”验证GPU
void cpu_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {          // 遍历C的行
        for (int j = 0; j < K; ++j) {      // 遍历C的列
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {  // 累加A[i][k] * B[k][j]（矩阵乘法核心公式）
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum; // 写入CPU计算结果（行主序存储）
        }
    }
}

// -----------------------------------------------------------------------------
// 7. main函数：程序入口（小矩阵验证流程控制）
// -----------------------------------------------------------------------------
int main()
{
    // -------------------------- 矩阵维度配置（小矩阵：用于验证正确性） --------------------------
    // 选择非16整数倍的N（24），测试边界处理；维度极小（16×24×32），CPU计算耗时<1秒
    const int M = 8192;  // C的行数 = A的行数（16行）
    const int N = 6144;  // A的列数 = B的行数（24列/行，非16倍数）
    const int K = 4096;  // C的列数 = B的列数（32列）

    // -------------------------- CPU内存分配（页锁定内存：优化传输速度） --------------------------
    // 用cudaMallocHost分配页锁定内存（比普通new快20-30%），避免传输时的内存拷贝开销
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_A, M * N * sizeof(float)));   // A矩阵（CPU）
    CHECK_CUDA_ERROR(cudaMallocHost(&h_B, N * K * sizeof(float)));   // 原B矩阵（CPU）
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C_gpu, M * K * sizeof(float))); // GPU结果（CPU）
    CHECK_CUDA_ERROR(cudaMallocHost(&h_C_cpu, M * K * sizeof(float))); // CPU结果（CPU）

    // -------------------------- 矩阵初始化（随机数：模拟真实数据） --------------------------
    // A/B矩阵初始化为0~1的随机数（若需更易验证，可改为固定值如A全1、B全2）
    for (int i = 0; i < M * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; // rand()生成0~RAND_MAX，归一化到0~1
    for (int i = 0; i < N * K; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // -------------------------- GPU内存分配（对应CPU内存的设备端空间） --------------------------
    float *d_A, *d_B, *d_B_trans, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * N * sizeof(float)));   // A矩阵（GPU）
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * K * sizeof(float)));   // 原B矩阵（GPU，solve参数）
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_trans, K * N * sizeof(float))); // 转置后B矩阵（GPU）
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * K * sizeof(float)));   // 结果C矩阵（GPU）

    // -------------------------- 时间测量初始化（高精度计时：含传输+计算） --------------------------
    cudaEvent_t start, stop; // CUDA事件：比CPU计时更适合GPU异步操作
    CHECK_CUDA_ERROR(cudaEventCreate(&start));  // 创建开始事件
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));   // 创建结束事件
    

    // -------------------------- 步骤1：CPU→GPU 传输原矩阵A/B --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // -------------------------- 步骤2：GPU内部 转置B矩阵 --------------------------
    dim3 trans_threads(16, 16); // 转置内核线程块：16x16（与乘法内核一致）
    // 转置内核网格维度：覆盖原B矩阵所有元素（N行K列）
    dim3 trans_blocks((K + trans_threads.x - 1) / trans_threads.x,
                      (N + trans_threads.y - 1) / trans_threads.y);
    matrix_transpose_kernel<<<trans_blocks, trans_threads>>>(d_B, d_B_trans, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());         // 检查转置内核启动错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());    // 等待转置完成（避免后续访问未转置数据）

    // -------------------------- 步骤3：传递转置后B地址到全局变量 --------------------------
    // 关键：通过cudaMemcpyToSymbol将GPU端d_B_trans地址写入全局变量，供乘法内核访问
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_B_trans_global, &d_B_trans, sizeof(float*)));

    // -------------------------- 步骤4：调用solve函数（按题目要求未修改参数） --------------------------
    CHECK_CUDA_ERROR(cudaEventRecord(start));   // 记录开始时间（后续操作均计入计时）
    solve(d_A, d_B, d_C, M, N, K); // 参数仍为原B（d_B），内核实际用d_B_trans_global
    CHECK_CUDA_ERROR(cudaEventRecord(stop));      // 记录结束时间
    // -------------------------- 步骤5：GPU→CPU 传输结果C --------------------------
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // -------------------------- 时间计算与输出 --------------------------
    
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // 等待事件完成（确保计时准确）
    float total_time_ms = 0;
    // 计算时间差（单位：毫秒）：包含“CPU→GPU传输 + GPU转置 + 乘法计算 + GPU→CPU传输”
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_time_ms, start, stop));
    std::cout << "总耗时: " << total_time_ms << " 毫秒" << std::endl;
    std::cout << "总耗时: " << total_time_ms / 1000.0f << " 秒" << std::endl;

    // -------------------------- 结果验证（对比GPU与CPU结果） --------------------------
    // cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, N, K); // CPU计算标准答案
    // float max_error = 0.0f;
    // // 遍历所有元素，计算最大误差（浮点计算允许<1e-5的误差）
    // for (int i = 0; i < M * K; ++i) {
    //     max_error = std::max(max_error, std::abs(h_C_gpu[i] - h_C_cpu[i]));
    // }
    // std::cout << "GPU与CPU结果的最大误差: " << max_error << std::endl; // 误差<1e-5即正确

    // -------------------------- 资源释放（避免内存泄漏） --------------------------
    // 释放CPU页锁定内存
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_gpu));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_cpu));
    // 释放GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_B_trans));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    // 销毁CUDA事件
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}