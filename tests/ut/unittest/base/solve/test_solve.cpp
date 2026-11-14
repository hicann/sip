/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "base_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {
// host fp32 高斯消元(带部分主元) golden: 解 A·X = B(与 sip_pta/交付套件同 fp32 标杆)
// bMats: B 的实际 batch(广播形态为 1, 各矩阵共享 B[0])
void HostSolveGolden(const std::vector<float>& a, const std::vector<float>& b, uint32_t n, uint32_t nrhs,
                     uint32_t bMats, std::vector<float>& x)
{
    uint32_t m = a.size() / (n * n);
    x.assign(m * n * nrhs, 0.0f);
    std::vector<float> aw(a);
    std::vector<float> bw(b);
    for (uint32_t mi = 0; mi < m; mi++) {
        float* A = aw.data() + mi * n * n;
        // B batch 广播: 各矩阵共享 B[0], 消元会原地改写 → 每矩阵先拷贝自己的 B 段
        std::vector<float> bSeg(bw.begin() + static_cast<int64_t>(mi % bMats) * n * nrhs,
                                bw.begin() + (static_cast<int64_t>(mi % bMats) + 1) * n * nrhs);
        float* B = bSeg.data();
        float* X = x.data() + mi * n * nrhs;
        for (uint32_t k = 0; k < n; k++) {
            uint32_t p = k;
            float best = std::fabs(A[k * n + k]);
            for (uint32_t r = k + 1; r < n; r++) {
                float v = std::fabs(A[r * n + k]);
                if (v > best) {
                    best = v;
                    p = r;
                }
            }
            if (p != k) {
                for (uint32_t c = 0; c < n; c++) {
                    std::swap(A[k * n + c], A[p * n + c]);
                }
                for (uint32_t c = 0; c < nrhs; c++) {
                    std::swap(B[k * nrhs + c], B[p * nrhs + c]);
                }
            }
            for (uint32_t r = k + 1; r < n; r++) {
                float f = A[r * n + k] / A[k * n + k];
                if (f == 0.0f) {
                    continue;
                }
                for (uint32_t c = k; c < n; c++) {
                    A[r * n + c] -= f * A[k * n + c];
                }
                for (uint32_t c = 0; c < nrhs; c++) {
                    B[r * nrhs + c] -= f * B[k * nrhs + c];
                }
            }
        }
        for (int32_t k = static_cast<int32_t>(n) - 1; k >= 0; k--) {
            for (uint32_t c = 0; c < nrhs; c++) {
                float s = B[k * nrhs + c];
                for (uint32_t j = k + 1; j < n; j++) {
                    s -= A[k * n + j] * X[j * nrhs + c];
                }
                X[k * nrhs + c] = s / A[k * n + k];
            }
        }
    }
}

struct CaseDesc {
    const char* name;
    uint32_t batch;
    uint32_t n;
    uint32_t nrhs;
    double mereLimit; // 大N case 放宽至 1e-3: 大 batch 随机矩阵存在条件数尾部
    bool bBcast;      // true: B 传 [1, n, nrhs] 广播形态 (kernel 模减偏移寻址, 不做展开)
};

// 需求 5 用例 + 大 N 用例 + 小 batch 负载均衡用例(全局块任务均分, batch < 核数)
const CaseDesc g_cases[] = {
    {"P0_[196,1000,2,2]x[196,1000,2,1]", 196000, 2, 1, 1e-4, false},
    {"P1_[196,4,4]x[196,4,4]", 196, 4, 4, 1e-4, false},
    {"P1_[196,3,3]x[1,3,16384]广播", 196, 3, 16384, 1e-4, true},
    {"P2_[196,3,3]x[196,3,3]", 196, 3, 3, 1e-4, false},
    {"P2_[196,3,3]x[1,3,1000]广播", 196, 3, 1000, 1e-4, true},
    {"广播_n2vec_[200,2,2]x[1,2,1]", 200, 2, 1, 1e-4, true}, // n=2 标量路径: 单核 batchNum=5 < 64, 未进 ProcessN2Vec
    {"广播_n2vec_[4096,2,2]x[1,2,2]", 4096, 2, 2, 1e-4,
     true}, // n=2 向量路径: 40 核下每核 batchNum=102, vecNum=64>0 进入 ProcessN2Vec
    {"广播_n2标量_[64,2,2]x[1,2,17]", 64, 2, 17, 1e-4, true},   // n=2 标量 chunk 路径
    {"广播_n96_[4,96,96]x[1,96,8]", 4, 96, 8, 1e-3, true},      // 分块 LU 路径(n>64)
    {"广播_n256_[2,256,256]x[1,256,4]", 2, 256, 4, 1e-3, true}, // 分块 LU 上限
    {"n16_[512,16,16]x[512,16,8]", 512, 16, 8, 1e-4, false},
    {"n32_[256,32,32]x[256,32,4]", 256, 32, 4, 1e-4, false},
    {"n64_[128,64,64]x[128,64,4]", 128, 64, 4, 1e-4, false},
    {"n64_[64,64,64]x[64,64,512]分块", 64, 64, 512, 1e-4, false},
    {"均衡_batch20_n3_nrhs1000", 20, 3, 1000, 1e-4, false},    // 40 块任务=核数, 每核恰 1 块
    {"均衡_batch16_n64_nrhs4096", 16, 64, 4096, 1e-4, false},  // 352 块, 每核 8~9 块
    {"均衡_n2大nrhs_fallback_LU", 64, 2, 100000, 1e-4, false}, // N2 超出 chunk 容量, 回落 LU 路径
    {"n2_奇nrhs_51_对齐回归", 4, 2, 51, 1e-4, false},    // 奇cols×奇chunk 回归: xBuf 非 32B 对齐曾触发 fault
    {"n2_奇nrhs_101_对齐回归", 91, 2, 101, 1e-4, false}, // 泛化测试发现的 X 全零缺陷(泛化 idx213)
    {"n2_奇nrhs_181_对齐回归", 57, 2, 181, 1e-4, false}, // 泛化测试发现的 X 全零缺陷(泛化 idx269)
    {"n2_奇nrhs_65_对齐回归", 8, 2, 65, 1e-4, false},    // 奇cols 边界
    {"大N_n96_[64,96,96]x[64,96,8]", 64, 96, 8, 1e-3, false},              // 分块 LU(1.5 panel)
    {"大N_n128_[32,128,128]x[32,128,4]", 32, 128, 4, 1e-3, false},         // 分块 LU(2 panel)
    {"大N_n256_[8,256,256]x[8,256,4]", 8, 256, 4, 1e-3, false},            // 分块 LU(4 panel, 上限)
    {"大N_n256_大cols_[4,256,256]x[4,256,300]", 4, 256, 300, 1e-3, false}, // 多列块 + 尾块
};

void RunSolveCase(const CaseDesc& cs, bool checkAccuracy, bool measurePerf)
{
    const uint32_t batch = cs.batch;
    const uint32_t n = cs.n;
    const uint32_t nrhs = cs.nrhs;
    const uint32_t bBatch = cs.bBcast ? 1U : batch; // B 广播形态仅 1 矩阵, kernel 共享 B[0]
    const size_t aElems = static_cast<size_t>(batch) * n * n;
    const size_t bElems = static_cast<size_t>(bBatch) * n * nrhs;
    const size_t xElems = static_cast<size_t>(batch) * n * nrhs; // X 输出恒为满 batch

    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<float> aHost(aElems);
    std::vector<float> bHost(bElems);
    for (size_t i = 0; i < aElems; i++) {
        aHost[i] = static_cast<float>(norm(rng));
    }
    for (uint32_t m = 0; m < batch; m++) {
        for (uint32_t d = 0; d < n; d++) {
            aHost[static_cast<size_t>(m) * n * n + d * n + d] += 3.0f; // 对角增强保证可逆
        }
    }
    for (size_t i = 0; i < bElems; i++) {
        bHost[i] = static_cast<float>(norm(rng));
    }

    std::vector<float> xHost(xElems, 0.0f);
    std::vector<int64_t> aShape = {static_cast<int64_t>(batch), static_cast<int64_t>(n), static_cast<int64_t>(n)};
    std::vector<int64_t> bShape = {static_cast<int64_t>(bBatch), static_cast<int64_t>(n), static_cast<int64_t>(nrhs)};
    std::vector<int64_t> xShape = {static_cast<int64_t>(batch), static_cast<int64_t>(n), static_cast<int64_t>(nrhs)};

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);

    aclTensor* aclA = nullptr;
    aclTensor* aclB = nullptr;
    aclTensor* aclX = nullptr;
    void* aDev = nullptr;
    void* bDev = nullptr;
    void* xDev = nullptr;
    ASSERT_EQ(CreateAclTensor(aHost, aShape, &aDev, aclDataType::ACL_FLOAT, &aclA), 0);
    ASSERT_EQ(CreateAclTensor(bHost, bShape, &bDev, aclDataType::ACL_FLOAT, &aclB), 0);
    ASSERT_EQ(CreateAclTensor(xHost, xShape, &xDev, aclDataType::ACL_FLOAT, &aclX), 0);

    AspbStatus st = solve(aclA, aclB, aclX, aclStream, nullptr);
    ASSERT_EQ(st, AsdSip::ErrorType::ACL_SUCCESS);
    MkiRtStreamSynchronize(stream);

    if (checkAccuracy) {
        ASSERT_EQ(
            aclrtMemcpy(xHost.data(), xElems * sizeof(float), xDev, xElems * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
            ::ACL_SUCCESS);
        std::vector<float> golden;
        HostSolveGolden(aHost, bHost, n, nrhs, bBatch, golden);

        double sumRel = 0.0;
        double maxRel = 0.0;
        double rms = 0.0;
        for (auto v : golden) {
            rms += static_cast<double>(v) * v;
        }
        rms = std::sqrt(rms / golden.size());
        double worstAbs = 0.0;
        double worstGold = 0.0;
        double worstDev = 0.0;
        for (size_t i = 0; i < xElems; i++) {
            double d = static_cast<double>(xHost[i]);
            double g = static_cast<double>(golden[i]);
            // 分母地板 3e-3·rms: |g| 低于该水位时相对误差已由分母噪声主导
            // (大N多列块 case 实测 worst 元素 |g|/rms≈8e-4 且 absErr/rms 仅 1.2e-4, 健康但
            //  MARE 被近零 golden 放大 —— 与 §5.4 n64×512 判据修正同类现象)
            double rel = std::fabs(d - g) / std::max(std::fabs(g), 3e-3 * rms);
            sumRel += rel;
            if (rel > maxRel) {
                maxRel = rel;
                worstAbs = std::fabs(d - g);
                worstGold = g;
                worstDev = d;
            }
        }
        double mere = sumRel / xElems;
        std::cout << "[solve][acc] " << cs.name << ": MERE=" << mere << " MARE=" << maxRel << " worst: dev=" << worstDev
                  << " gold=" << worstGold << " absErr=" << worstAbs << " absErr/rms=" << worstAbs / rms << std::endl;
        EXPECT_LT(mere, cs.mereLimit);
        EXPECT_LT(maxRel, 1e-1);
    }

    if (measurePerf) {
        for (int i = 0; i < 10; i++) {
            ASSERT_EQ(solve(aclA, aclB, aclX, aclStream, nullptr), AsdSip::ErrorType::ACL_SUCCESS);
        }
        MkiRtStreamSynchronize(stream);
        auto t0 = std::chrono::steady_clock::now();
        constexpr int iters = 200;
        for (int i = 0; i < iters; i++) {
            solve(aclA, aclB, aclX, aclStream, nullptr);
        }
        MkiRtStreamSynchronize(stream);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "[solve][perf] " << cs.name << ": pipelined = " << us << " us/call" << std::endl;
    }

    aclDestroyTensor(aclA);
    aclDestroyTensor(aclB);
    aclDestroyTensor(aclX);
    aclrtFree(aDev);
    aclrtFree(bDev);
    aclrtFree(xDev);
    MkiRtStreamSynchronize(stream);
    TensorContext emptyCtx;
    OpTestEnd(deviceId, emptyCtx, stream);
}
} // namespace

TEST(TestOpSolve, SolveAccuracyAllShapes)
{
    for (const auto& cs : g_cases) {
        RunSolveCase(cs, true, false);
    }
}

TEST(TestOpSolve, SolvePerfAllShapes)
{
    for (const auto& cs : g_cases) {
        RunSolveCase(cs, false, true);
    }
}
