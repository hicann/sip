/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdBlasStrmv 负向参数校验用例（issue #125/#139）：
// 非法参数应被入口拦截并返回错误码，而非静默按默认值计算或抛异常

#include <gtest/gtest.h>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "blas_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;
using namespace Mki;

namespace {

// issue #125：incx!=1 / lda!=n / uplo=FULL / incx<=0 均应返回 ACL_ERROR_INVALID_PARAM
TEST(TestBlasStrmvNegative, ParamValidation)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t n = 4;
    std::vector<float> xHost(n, 1.0f);
    std::vector<float> aHost(n * n, 1.0f);
    aclTensor* aclX = nullptr;
    aclTensor* aclA = nullptr;
    void* xAddr = nullptr;
    void* aAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(xHost, std::vector<int64_t>{n}, &xAddr, aclDataType::ACL_FLOAT, &aclX), 0);
    ASSERT_EQ(CreateAclTensor(aHost, std::vector<int64_t>{n, n}, &aAddr, aclDataType::ACL_FLOAT, &aclA), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(
        asdBlasMakeStrmvPlan(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N, n),
        AsdSip::ErrorType::ACL_SUCCESS);

    // incx != 1
    EXPECT_EQ(asdBlasStrmv(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N,
                           asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, n, aclA, n, aclX, 2),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // incx <= 0
    EXPECT_EQ(asdBlasStrmv(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N,
                           asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, n, aclA, n, aclX, 0),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // lda != n
    EXPECT_EQ(asdBlasStrmv(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N,
                           asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, n, aclA, n + 1, aclX, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // uplo = FULL
    EXPECT_EQ(asdBlasStrmv(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_FULL, asdBlasOperation_t::ASDBLAS_OP_N,
                           asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, n, aclA, n, aclX, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclX);
    aclDestroyTensor(aclA);
    aclrtFree(xAddr);
    aclrtFree(aAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// issue #139：n=1 且 A 为 0 维张量时，应被维度校验拦截而非 tiling 层抛 std::out_of_range
TEST(TestBlasStrmvNegative, ZeroDimTensorRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t n = 1;
    std::vector<float> xHost(n, 1.0f);
    aclTensor* aclX = nullptr;
    void* xAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(xHost, std::vector<int64_t>{n}, &xAddr, aclDataType::ACL_FLOAT, &aclX), 0);
    void* aAddr = nullptr;
    ASSERT_EQ(aclrtMalloc(&aAddr, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ::ACL_SUCCESS);
    // 0 维（标量）张量：dims 为空
    aclTensor* aclA = aclCreateTensor(nullptr, 0, aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr,
                                      0, aAddr);
    ASSERT_NE(aclA, nullptr);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(
        asdBlasMakeStrmvPlan(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N, n),
        AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdBlasStrmv(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, asdBlasOperation_t::ASDBLAS_OP_N,
                           asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, n, aclA, n, aclX, 1),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclX);
    aclDestroyTensor(aclA);
    aclrtFree(xAddr);
    aclrtFree(aAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

} // namespace
