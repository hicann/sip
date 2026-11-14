/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ops 层 FFT InferShape 空 dims 负向用例（issue #397）：
// 1) 直调 operation->InferShape：0 维（空 dims）输入触发新增的 dims.empty() 防护，
//    返回 InferShape 失败状态而非 dims[0] 越界读（覆盖 FftC2R/FftR2C/DftC2R/DftR2C
//    四个 910B 可达 op；FftC2CArch35 的 dims[0] 读取仅在 950 分支，910B 不可达）；
// 2) C API 直调 0 维输入：asdFftExecC2R/asdFftExecR2R 的 plan match 层先拦截，
//    断言不崩溃且返回 ACL_ERROR_INVALID_PARAM。

#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "base_api.h"
#include "fft_api.h"
#include "params/fft_all_mix.h"
#include "params/dft_c2r.h"
#include "params/dft_r2c.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;
using namespace Mki;

namespace {

// 0 维（标量）aclTensor：shape/dims 均为空（issue #343 用例同款构造）
int CreateScalarAclTensor(aclDataType dtype, size_t byteSize, aclTensor** tensor, void** addr)
{
    auto ret = aclrtMalloc(addr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ::ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    *tensor = aclCreateTensor(nullptr, 0, dtype, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0, *addr);
    return *tensor != nullptr ? 0 : -1;
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

// 直调 operation->InferShape：0 维输入应被新增守卫拒绝（返回失败状态而非越界读崩溃）
void RunInferShapeWithZeroDim(const char* opName, const Any& specificParam, aclDataType inDtype)
{
    Mki::Operation* op = Ops::Instance().GetOperationByName(opName);
    ASSERT_NE(op, nullptr) << "operation not found: " << opName;

    aclTensor* aclIn = nullptr;
    aclTensor* aclOut = nullptr;
    void* inAddr = nullptr;
    void* outAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(inDtype, sizeof(float), &aclIn, &inAddr), 0);
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &aclOut, &outAddr), 0);

    LaunchParam launch;
    launch.SetParam(specificParam);
    launch.AddInTensor(aclIn);
    launch.AddOutTensor(aclOut);

    const Status status = op->InferShape(launch);
    EXPECT_FALSE(status.Ok()) << "empty-dims input should be rejected by " << opName;

    DestroyAclTensorWithAddr(aclIn, inAddr);
    DestroyAclTensorWithAddr(aclOut, outAddr);
}

} // namespace

TEST(TestFftInferShapeZeroDim, FftC2ROperation)
{
    MkiRtStream stream = OpTestInit(0);
    OpParam::FftAllMix param;
    RunInferShapeWithZeroDim("FftC2ROperation", param, aclDataType::ACL_COMPLEX64);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestFftInferShapeZeroDim, FftR2COperation)
{
    MkiRtStream stream = OpTestInit(0);
    OpParam::FftAllMix param;
    RunInferShapeWithZeroDim("FftR2COperation", param, aclDataType::ACL_FLOAT);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestFftInferShapeZeroDim, DftC2ROperation)
{
    MkiRtStream stream = OpTestInit(0);
    OpParam::DftC2R param;
    RunInferShapeWithZeroDim("DftC2ROperation", param, aclDataType::ACL_COMPLEX64);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestFftInferShapeZeroDim, DftR2COperation)
{
    MkiRtStream stream = OpTestInit(0);
    OpParam::DftR2C param;
    RunInferShapeWithZeroDim("DftR2COperation", param, aclDataType::ACL_FLOAT);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

// C API 直调 0 维输入：plan match 层先拦截（输入维度数 < plan 维度数），不崩溃且显式报错
TEST(TestFftInferShapeZeroDim, CApiExecC2RRejectZeroDim)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inAddr = nullptr;
    void* outAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_COMPLEX64, sizeof(float), &aclInput, &inAddr), 0);
    std::vector<float> outHost(8, 0.0f);
    ASSERT_EQ(CreateAclTensor(outHost, {8}, &outAddr, aclDataType::ACL_FLOAT, &aclOutput), 0);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 16, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    ASSERT_EQ(asdFftSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftExecC2R(handle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdFftDestroy(handle);
    DestroyAclTensorWithAddr(aclInput, inAddr);
    DestroyAclTensorWithAddr(aclOutput, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}

TEST(TestFftInferShapeZeroDim, CApiExecR2CRejectZeroDim)
{
    MkiRtStream stream = OpTestInit(0);
    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inAddr = nullptr;
    void* outAddr = nullptr;
    ASSERT_EQ(CreateScalarAclTensor(aclDataType::ACL_FLOAT, sizeof(float), &aclInput, &inAddr), 0);
    std::vector<std::complex<float>> outHost(8, {0.0f, 0.0f});
    ASSERT_EQ(CreateAclTensor(outHost, {8}, &outAddr, aclDataType::ACL_COMPLEX64, &aclOutput), 0);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 16, asdFftType::ASCEND_FFT_R2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    ASSERT_EQ(asdFftSetStream(handle, aclStream), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftExecR2C(handle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdFftDestroy(handle);
    DestroyAclTensorWithAddr(aclInput, inAddr);
    DestroyAclTensorWithAddr(aclOutput, outAddr);
    TensorContext context;
    OpTestEnd(0, context, stream);
}
