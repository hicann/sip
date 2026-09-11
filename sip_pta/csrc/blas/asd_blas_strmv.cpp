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
#include <vector>

namespace sip_pta {

/**
 * @brief 三角矩阵-向量乘法算子 (Strmv): x = op(A) * x (原地更新)
 */
at::Tensor asdBlasStrmv(const at::Tensor& A, at::Tensor& x, int64_t uplo, int64_t trans, int64_t diag)
{
    // 基本校验：算子明确要求 FLOAT32，且 A 为方阵，x 长度需匹配
    TORCH_CHECK(A.scalar_type() == at::kFloat && x.scalar_type() == at::kFloat,
                "asd_blas_strmv: Tensors A and x must be of type Float32.");
    TORCH_CHECK(A.dim() == sip_pta::DIM_2 && A.size(0) == A.size(1), "asd_blas_strmv: Matrix A must be square.");
    TORCH_CHECK(x.size(0) == A.size(0), "asd_blas_strmv: Dimensions of A and x must match.");
    c10::DeviceGuard guard(A.device());
    // 参数准备
    AsdSip::asdBlasFillMode_t uploEnum = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
    AsdSip::asdBlasOperation_t transEnum = static_cast<AsdSip::asdBlasOperation_t>(trans);
    AsdSip::asdBlasDiagType_t diagEnum = static_cast<AsdSip::asdBlasDiagType_t>(diag);

    int64_t n = A.size(0);
    int64_t lda = n;
    int64_t incx = 1;

    std::vector<int64_t> params = {static_cast<int64_t>(uploEnum), static_cast<int64_t>(transEnum), n};

    // 封装 MakePlan 函数逻辑
    auto makePlan = [uploEnum, transEnum, n](AsdSip::asdBlasHandle handle) {
        return AsdSip::asdBlasMakeStrmvPlan(handle, uploEnum, transEnum, n);
    };

    EXEC_BLAS_FUNC(AsdSip::asdBlasStrmv, makePlan, params, uploEnum, transEnum, diagEnum, n, A, lda, x, incx);

    return x;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_strmv(Tensor A, Tensor(a!) x, int uplo, int trans, int diag) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_strmv", &asdBlasStrmv); }

} // namespace sip_pta
