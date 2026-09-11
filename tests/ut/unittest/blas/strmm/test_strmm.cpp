/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdBlasStrmm 负向校验用例（issue #133）：
// 输出张量 C 元素数不满足 m*n 时应被拦截，而非 kernel 越界写

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

// issue #133：C 元素数 < m*n 应返回 ACL_ERROR_OP_INPUT_NOT_MATCH
TEST(TestBlasStrmmNegative, OutputShapeValidation)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t m = 4;
    const int64_t n = 4;
    std::vector<float> aHost(m * m, 1.0f);
    std::vector<float> bHost(m * n, 1.0f);
    std::vector<float> cHost(m * n - 1, 0.0f); // 元素数 15 < m*n=16
    aclTensor* aclA = nullptr;
    aclTensor* aclB = nullptr;
    aclTensor* aclC = nullptr;
    void* aAddr = nullptr;
    void* bAddr = nullptr;
    void* cAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(aHost, std::vector<int64_t>{m, m}, &aAddr, aclDataType::ACL_FLOAT, &aclA), 0);
    ASSERT_EQ(CreateAclTensor(bHost, std::vector<int64_t>{m, n}, &bAddr, aclDataType::ACL_FLOAT, &aclB), 0);
    ASSERT_EQ(CreateAclTensor(cHost, std::vector<int64_t>{m * n - 1}, &cAddr, aclDataType::ACL_FLOAT, &aclC), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeStrmmPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdBlasStrmm(handle, asdBlasSideMode_t::ASDBLAS_SIDE_LEFT, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER,
                           asdBlasOperation_t::ASDBLAS_OP_N, asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, m, n, 1.0f, aclA,
                           m, aclB, n, aclC, n),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclB);
    aclDestroyTensor(aclC);
    aclrtFree(aAddr);
    aclrtFree(bAddr);
    aclrtFree(cAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

} // namespace
