/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// asdBlasCmatinvBatched 负向校验用例（issue #138）：
// lda/lda_inv 非法（<=0 或 != n）时应返回 ACL_ERROR_INVALID_PARAM，而非静默按连续布局计算

#include <gtest/gtest.h>
#include <complex>
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

// issue #138：lda<=0 / lda_inv<=0 / lda!=n 均应返回 ACL_ERROR_INVALID_PARAM
TEST(TestBlasCmatinvBatchedNegative, LdaValidation)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t n = 4;
    const int64_t batchSize = 2;
    std::vector<std::complex<float>> aHost(batchSize * n * n, {1.0f, 0.0f});
    std::vector<std::complex<float>> ainvHost(batchSize * n * n, {0.0f, 0.0f});
    aclTensor* aclA = nullptr;
    aclTensor* aclAinv = nullptr;
    void* aAddr = nullptr;
    void* ainvAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(aHost, std::vector<int64_t>{batchSize, n, n}, &aAddr, aclDataType::ACL_COMPLEX64, &aclA),
              0);
    ASSERT_EQ(CreateAclTensor(ainvHost, std::vector<int64_t>{batchSize, n, n}, &ainvAddr, aclDataType::ACL_COMPLEX64,
                              &aclAinv),
              0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeCmatinvBatchedPlan(handle, n, batchSize), AsdSip::ErrorType::ACL_SUCCESS);

    // lda <= 0
    EXPECT_EQ(asdBlasCmatinvBatched(handle, n, aclA, 0, aclAinv, n, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // lda_inv <= 0
    EXPECT_EQ(asdBlasCmatinvBatched(handle, n, aclA, n, aclAinv, -1, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // lda != n
    EXPECT_EQ(asdBlasCmatinvBatched(handle, n, aclA, 2 * n, aclAinv, 2 * n, nullptr, batchSize),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclA);
    aclDestroyTensor(aclAinv);
    aclrtFree(aAddr);
    aclrtFree(ainvAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

} // namespace
