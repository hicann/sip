/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "blas_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 批量复数矩阵求逆算子 (CmatinvBatched)
 */
at::Tensor asdBlasCmatinvBatched(const at::Tensor& A)
{
    // 基本校验：支持 ComplexFloat 和 ComplexHalf
    TORCH_CHECK((A.scalar_type() == at::kComplexFloat || A.scalar_type() == at::kComplexHalf),
                "asd_blas_cmatinv_batched: Tensor must be ComplexFloat or ComplexHalf.");

    // 维度校验：必须为 3D 张量且后两维相等（方阵）
    TORCH_CHECK(A.dim() == sip_pta::DIM_3 && A.size(1) == A.size(sip_pta::DIM_2),
                "asd_blas_cmatinv_batched: Tensor must be a 3D square matrix [Batch, N, N].");

    c10_npu::NPUGuard guard(A.device());
    // 参数准备
    int64_t batchSize = A.size(0);
    int64_t n = A.size(1);

    // 依据算子定义：lda 和 lda_inv 仅支持等于 n
    int64_t lda = n;
    int64_t lda_inv = n;

    // 输出空间开辟：逆矩阵 Ainv 和状态返回数组 info
    at::Tensor Ainv = at::empty_like(A);
    at::Tensor info; // 实际不生效

    // 构建传入 getBlasHandle 的 planParam 参数
    std::vector<int64_t> params = {n, batchSize};
    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    aclTensor* acl_a_inv = CreateAclTensorFromAtTensor(Ainv);
    // 根据数据类型分发调用对应精度算子
    if (A.scalar_type() == at::kComplexFloat) {
        // 单精度逻辑 (Complex64)
        auto makePlan = [n, batchSize](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeCmatinvBatchedPlan(handle, n, batchSize);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasCmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    } else {
        // 半精度逻辑 (Complex32)
        auto makePlan = [n, batchSize](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeHCmatinvBatchedPlan(handle, n, batchSize);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasHCmatinvBatched, makePlan, params, n, acl_a, lda, acl_a_inv, lda_inv, info,
                       batchSize);
    }

    return Ainv;
}

// 算子注册
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_cmatinv_batched(Tensor A) -> Tensor"); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cmatinv_batched", &asdBlasCmatinvBatched); }
} // namespace sip_pta
