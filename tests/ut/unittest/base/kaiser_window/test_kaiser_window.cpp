/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdKaiserWindow host C API 负向与边界用例（issue #396）：!285 合入时 base 目录零
// UT 覆盖，本文件按 base 算子测试惯例补齐：
//   - 全部参数校验分支的对应错误码（null/length/dtype/format/shape/stride/stream）
//   - windowLength=0/1/INT32_MAX 边界（INT32_MAX 以"边界值通过 + 下一校验分支报错"
//     方式验证，避免 8GB 输出分配）
//   - 非零 storage offset 视图分支（PTA 的 at::empty 输出恒走别名分支，此处直调
//     C API 构造偏移视图，验证视图创建与写回位置正确）
//   - 缓存命中路径与首跑路径输出一致性（thread_local 缓存按 (context, param) 键控）

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "base_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;

namespace {

constexpr int64_t KAISER_LEN = 17;

// 修正第一类贝塞尔函数 I0(x) 级数（与 torch.kaiser_window 的 float64 golden 对齐）
double BesselI0(double x)
{
    const double halfX2 = x * x / 4.0;
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 60; ++k) {
        term *= halfX2 / (k * k);
        sum += term;
        if (term < 1e-18 * sum) {
            break;
        }
    }
    return sum;
}

// torch.kaiser_window 语义: periodic=True 时 α=length/2, 否则 α=(length-1)/2;
// w[i] = I0(β·sqrt(1-((i-α)/α)²)) / I0(β); length<=1 时恒为 1
std::vector<double> KaiserGolden(int64_t length, bool periodic, double beta)
{
    std::vector<double> w(static_cast<size_t>(length), 1.0);
    if (length <= 1) {
        return w;
    }
    const double alpha = periodic ? length / 2.0 : (length - 1) / 2.0;
    const double i0Beta = BesselI0(beta);
    for (int64_t i = 0; i < length; ++i) {
        const double t = (i - alpha) / alpha;
        const double r = std::sqrt(std::max(0.0, 1.0 - t * t));
        w[static_cast<size_t>(i)] = BesselI0(beta * r) / i0Beta;
    }
    return w;
}

bool CompareClose(const std::vector<float>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > atol + rtol * std::fabs(expected[i])) {
            LOG_PRINT("kaiser mismatch at %zu: actual=%.9f expected=%.9f\n", i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

// IEEE half → float 位级转换（fp16 视图用例读回用）
float HalfBitsToFloat(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t man = bits & 0x3FFu;
    uint32_t fbits;
    if (exp == 0) {
        fbits = sign | (man << 13);
    } else if (exp == 0x1F) {
        fbits = sign | 0x7F800000u | (man << 13);
    } else {
        fbits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &fbits, sizeof(out));
    return out;
}

void DestroyAclTensorWithAddr(aclTensor* tensor, void* addr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (addr != nullptr) {
        aclrtFree(addr);
    }
}

std::vector<float> ReadBack(aclTensor* tensor, void* addr, size_t elems)
{
    std::vector<float> host(elems, 0.0f);
    const size_t bytes = elems * sizeof(float);
    auto ret = aclrtMemcpy(host.data(), bytes, addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(ret, ::ACL_SUCCESS);
    return host;
}

} // namespace

// ---- 参数校验分支（issue #396: core/base/kaiser_window.cpp:24-49 错误码路径）----

TEST(TestKaiserWindowParamGuard, RejectNullOutput)
{
    MkiRtStream stream = OpTestInit(0);
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    EXPECT_EQ(asdKaiserWindow(nullptr, KAISER_LEN, true, 12.0, aclStream, nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectNegativeLength)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(KAISER_LEN, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, -1, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectLengthOverInt32Max)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(KAISER_LEN, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, static_cast<int64_t>(INT32_MAX) + 1, true, 12.0, static_cast<aclrtStream>(stream),
                              nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// INT32_MAX 上边界：长度校验应放行，由后续 dtype 校验分支报错（避免 8GB 输出分配）
TEST(TestKaiserWindowParamGuard, Int32MaxBoundaryPassesLengthCheck)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<int32_t> host(KAISER_LEN, 0);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_INT32, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, INT32_MAX, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectUnsupportedDtype)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<int32_t> host(KAISER_LEN, 0);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_INT32, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectNonNdFormat)
{
    MkiRtStream stream = OpTestInit(0);
    std::vector<float> host(KAISER_LEN, 0.0f);
    std::vector<int64_t> shape = {KAISER_LEN};
    std::vector<int64_t> strides = {1};
    void* outAddr = nullptr;
    auto ret = aclrtMalloc(&outAddr, static_cast<int64_t>(host.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclTensor* out = aclCreateTensor(shape.data(), 1, aclDataType::ACL_FLOAT, strides.data(), 0,
                                     aclFormat::ACL_FORMAT_NCHW, shape.data(), 1, outAddr);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectShapeMismatch)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(8, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {8}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectNonContiguousOutput)
{
    MkiRtStream stream = OpTestInit(0);
    std::vector<float> host(31, 0.0f);
    std::vector<int64_t> shape = {KAISER_LEN};
    std::vector<int64_t> strides = {2}; // 跨步视图: 非连续
    void* outAddr = nullptr;
    auto ret = aclrtMalloc(&outAddr, static_cast<int64_t>(host.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclTensor* out = aclCreateTensor(shape.data(), 1, aclDataType::ACL_FLOAT, strides.data(), 0,
                                     aclFormat::ACL_FORMAT_ND, shape.data(), 1, outAddr);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectNullStream)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(KAISER_LEN, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    EXPECT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, nullptr, nullptr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowParamGuard, RejectNullWorkspaceSizePtr)
{
    EXPECT_EQ(asdKaiserWindowGetWorkspaceSize(nullptr), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

// ---- 边界长度（issue #396: windowLength=0/1/INT32_MAX）----

TEST(TestKaiserWindowBoundary, ZeroLengthReturnsSuccess)
{
    MkiRtStream stream = OpTestInit(0);
    std::vector<int64_t> shape = {0};
    std::vector<int64_t> strides = {1};
    void* outAddr = nullptr;
    auto ret = aclrtMalloc(&outAddr, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclTensor* out = aclCreateTensor(shape.data(), 1, aclDataType::ACL_FLOAT, strides.data(), 0,
                                     aclFormat::ACL_FORMAT_ND, shape.data(), 1, outAddr);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(asdKaiserWindow(out, 0, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowBoundary, OneLengthIsOne)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(1, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {1}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    ASSERT_EQ(asdKaiserWindow(out, 1, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
    std::vector<float> actual = ReadBack(out, outAddr, 1);
    EXPECT_EQ(actual[0], 1.0f);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ---- 视图创建分支（issue #396: :83-89, PTA 路径永不覆盖的非零 offset 分支）----

TEST(TestKaiserWindowViewBranch, NonZeroOffsetFloatView)
{
    MkiRtStream stream = OpTestInit(0);
    const int64_t offset = 2;
    const int64_t storageLen = KAISER_LEN + offset;
    std::vector<float> host(storageLen, 7.5f); // 前后哨兵: 窗口只应写 [2, 2+L)
    std::vector<int64_t> viewShape = {KAISER_LEN};
    std::vector<int64_t> viewStrides = {1};
    std::vector<int64_t> storageShape = {storageLen};
    void* outAddr = nullptr;
    auto ret = aclrtMalloc(&outAddr, static_cast<int64_t>(host.size() * sizeof(float)), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    ret = aclrtMemcpy(outAddr, host.size() * sizeof(float), host.data(),
                      static_cast<int64_t>(host.size() * sizeof(float)), ACL_MEMCPY_HOST_TO_DEVICE);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclTensor* out = aclCreateTensor(viewShape.data(), 1, aclDataType::ACL_FLOAT, viewStrides.data(), offset,
                                     aclFormat::ACL_FORMAT_ND, storageShape.data(), 1, outAddr);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
    std::vector<float> actual = ReadBack(out, outAddr, static_cast<size_t>(storageLen));
    std::vector<double> golden = KaiserGolden(KAISER_LEN, true, 12.0);
    EXPECT_TRUE(CompareClose(std::vector<float>(actual.begin() + offset, actual.end()), golden, 2e-4, 2e-4));
    EXPECT_EQ(actual[0], 7.5f); // 偏移前哨兵不被写坏
    EXPECT_EQ(actual[1], 7.5f);
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowViewBranch, NonZeroOffsetHalfView)
{
    MkiRtStream stream = OpTestInit(0);
    const int64_t offset = 3;
    const int64_t storageLen = KAISER_LEN + offset;
    std::vector<int64_t> viewShape = {KAISER_LEN};
    std::vector<int64_t> viewStrides = {1};
    std::vector<int64_t> storageShape = {storageLen};
    void* outAddr = nullptr;
    auto ret = aclrtMalloc(&outAddr, static_cast<int64_t>(storageLen * sizeof(uint16_t)), ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    aclTensor* out = aclCreateTensor(viewShape.data(), 1, aclDataType::ACL_FLOAT16, viewStrides.data(), offset,
                                     aclFormat::ACL_FORMAT_ND, storageShape.data(), 1, outAddr);
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 12.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
    // half 视图读回: 视图分支按 uint16 itemSize 计算偏移地址
    std::vector<uint16_t> bits(storageLen, 0);
    ret = aclrtMemcpy(bits.data(), bits.size() * sizeof(uint16_t), outAddr,
                      static_cast<int64_t>(bits.size() * sizeof(uint16_t)), ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    std::vector<double> golden = KaiserGolden(KAISER_LEN, true, 12.0);
    for (int64_t i = 0; i < KAISER_LEN; ++i) {
        const float actual = HalfBitsToFloat(bits[static_cast<size_t>(offset + i)]);
        EXPECT_NEAR(actual, static_cast<float>(golden[static_cast<size_t>(i)]), 2e-3) << "half view mismatch at " << i;
    }
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// ---- 数值与缓存路径（issue #396: 缓存命中与首跑输出一致）----

void RunKaiserAndCheck(MkiRtStream stream, int64_t length, bool periodic, double beta, double atol, double rtol)
{
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(static_cast<size_t>(length), 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {length}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    ASSERT_EQ(asdKaiserWindow(out, length, periodic, beta, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
    std::vector<float> actual = ReadBack(out, outAddr, static_cast<size_t>(length));
    EXPECT_TRUE(CompareClose(actual, KaiserGolden(length, periodic, beta), atol, rtol));
    DestroyAclTensorWithAddr(out, outAddr);
}

TEST(TestKaiserWindowValues, PeriodicBeta12)
{
    MkiRtStream stream = OpTestInit(0);
    RunKaiserAndCheck(stream, KAISER_LEN, true, 12.0, 2e-4, 2e-4);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowValues, SymmetricBeta12)
{
    MkiRtStream stream = OpTestInit(0);
    RunKaiserAndCheck(stream, 16, false, 12.0, 2e-4, 2e-4);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestKaiserWindowValues, ZeroBetaIsRectangular)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* out = nullptr;
    void* outAddr = nullptr;
    std::vector<float> host(KAISER_LEN, 0.0f);
    ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
    ASSERT_EQ(asdKaiserWindow(out, KAISER_LEN, true, 0.0, static_cast<aclrtStream>(stream), nullptr),
              AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
    std::vector<float> actual = ReadBack(out, outAddr, static_cast<size_t>(KAISER_LEN));
    for (float v : actual) {
        EXPECT_NEAR(v, 1.0f, 2e-4);
    }
    DestroyAclTensorWithAddr(out, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// 缓存命中路径与首跑路径输出一致（同 param 第二次调用走 thread_local 缓存命中）
TEST(TestKaiserWindowCache, CacheHitConsistentWithFirstRun)
{
    MkiRtStream stream = OpTestInit(0);

    auto runOnce = [stream](double beta, std::vector<float>& sink) {
        aclTensor* out = nullptr;
        void* outAddr = nullptr;
        std::vector<float> host(KAISER_LEN, 0.0f);
        ASSERT_EQ(CreateAclTensor(host, {KAISER_LEN}, &outAddr, aclDataType::ACL_FLOAT, &out), 0);
        ASSERT_EQ(asdKaiserWindow(out, KAISER_LEN, true, beta, static_cast<aclrtStream>(stream), nullptr),
                  AsdSip::ErrorType::ACL_SUCCESS);
        ASSERT_EQ(MkiRtStreamSynchronize(stream), 0);
        sink = ReadBack(out, outAddr, static_cast<size_t>(KAISER_LEN));
        DestroyAclTensorWithAddr(out, outAddr);
    };

    std::vector<float> firstRun;
    std::vector<float> cacheHit;
    std::vector<float> otherBeta;
    RunKaiserAndCheck(stream, KAISER_LEN, true, 12.0, 2e-4, 2e-4); // 首跑: 填充缓存
    runOnce(12.0, cacheHit);                                       // 缓存命中
    RunKaiserAndCheck(stream, KAISER_LEN, true, 4.5, 2e-4, 2e-4);  // 换 beta: 缓存失效重取
    runOnce(4.5, otherBeta);                                       // 缓存命中 beta=4.5
    RunKaiserAndCheck(stream, KAISER_LEN, true, 12.0, 2e-4, 2e-4); // beta 换回: 再一次缓存重建
    runOnce(12.0, firstRun);

    ASSERT_EQ(firstRun.size(), static_cast<size_t>(KAISER_LEN));
    ASSERT_EQ(cacheHit.size(), firstRun.size());
    ASSERT_EQ(otherBeta.size(), firstRun.size());
    for (size_t i = 0; i < firstRun.size(); ++i) {
        EXPECT_EQ(firstRun[i], cacheHit[i]) << "cache-hit output differs at " << i;
    }
    bool differs = false;
    for (size_t i = 0; i < firstRun.size(); ++i) {
        if (std::fabs(firstRun[i] - otherBeta[i]) > 1e-6f) {
            differs = true;
        }
    }
    EXPECT_TRUE(differs) << "different beta should produce a different window";

    TensorContext context;
    OpTestEnd(0, context, stream);
}
