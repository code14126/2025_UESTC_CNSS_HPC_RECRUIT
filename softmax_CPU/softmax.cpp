#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <cstdlib>  // for rand/srand
#include <iomanip>  // for setprecision

// 单线程版本
void vanilla_softmax(const std::vector<float>& input, std::vector<float>& output) {
    int N = input.size();
    if (N == 0) return;

    // 步骤1：计算最大值（数值稳定）
    float max_val = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // 步骤2：计算指数并累加和
    float sum_exp = 0.0f;
    output.resize(N);
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val);  // 减去最大值避免溢出
        sum_exp += output[i];
    }

    // 步骤3：归一化
    for (int i = 0; i < N; ++i) {
        output[i] /= sum_exp;
    }
}

// 多线程优化版本（OpenMP）
void omp_softmax(const std::vector<float>& input, std::vector<float>& output) {
    int N = input.size();
    if (N == 0) return;

    // 步骤1：并行计算最大值（reduction确保线程安全）
    float max_val = input[0];
#pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < N; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // 步骤2：并行计算指数+累加和（reduction求和）
    float sum_exp = 0.0f;
    output.resize(N);
#pragma omp parallel for reduction(+:sum_exp)
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // 步骤3：并行归一化（无依赖，直接并行）
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        output[i] /= sum_exp;
    }
}

// 生成测试数据（范围[-10.0, 10.0]）
void generate_input(std::vector<float>& input, int N) {
    input.resize(N);
    std::srand(42);  // 固定种子，确保结果可复现
    for (int i = 0; i < N; ++i) {
        input[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
    }
}

// 验证结果正确性（允许浮点误差）
bool check_correctness(const std::vector<float>& ref, const std::vector<float>& test, float eps = 1e-5f) {
    if (ref.size() != test.size()) return false;
    int N = ref.size();
    for (int i = 0; i < N; ++i) {
        if (std::fabs(ref[i] - test[i]) > eps) {
            std::cerr << "验证失败：索引 " << i << "，参考值 " << ref[i] 
                      << "，测试值 " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 5000000;  // 5e6元素，符合需求
    std::vector<float> input, output_vanilla, output_omp;

    // 生成输入数据
    generate_input(input, N);
    std::cout << "数据生成完成，大小：" << N << " 元素" << std::endl;

    // 单线程版本计时
    auto start_vanilla = std::chrono::high_resolution_clock::now();
    vanilla_softmax(input, output_vanilla);
    auto end_vanilla = std::chrono::high_resolution_clock::now();
    double time_vanilla = std::chrono::duration<double>(end_vanilla - start_vanilla).count();
    std::cout << "单线程耗时：" << std::fixed << std::setprecision(4) << time_vanilla << " 秒" << std::endl;

    // 多线程版本计时
    auto start_omp = std::chrono::high_resolution_clock::now();
    omp_softmax(input, output_omp);
    auto end_omp = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double>(end_omp - start_omp).count();
    std::cout << "多线程耗时：" << std::fixed << std::setprecision(4) << time_omp << " 秒" << std::endl;

    // 验证正确性
    if (check_correctness(output_vanilla, output_omp)) {
        std::cout << "结果验证通过（误差在允许范围内）" << std::endl;
    } else {
        std::cerr << "结果验证失败" << std::endl;
        return 1;
    }

    // 性能对比
    double ratio = time_omp / time_vanilla;
    std::cout << "多线程耗时 / 单线程耗时 = " << std::fixed << std::setprecision(2) << (ratio * 100) << "%" << std::endl;
    if (ratio < 0.7) {
        std::cout << "满足性能要求（<70%）" << std::endl;
    } else {
        std::cout << "不满足性能要求（≥70%）" << std::endl;
    }

    return 0;
}