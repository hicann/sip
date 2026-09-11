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
 * @brief 单精度复数矩阵向量乘法算子 (Ctrmv)
 * 计算公式: x = op(A) * x
 */
at::Tensor asdBlasCtrmv(const at::Tensor& A, at::Tensor& x, int64_t uplo, int64_t trans, int64_t diag)
{
    // 基本数据类型校验
    // 根据算子定义，目前仅支持 ComplexFloat 单精度复数
    TORCH_CHECK(A.scalar_type() == at::kComplexFloat && x.scalar_type() == at::kComplexFloat,
                "asd_blas_ctrmv: Tensors must be ComplexFloat (Complex64).");

    // 维度校验
    // 矩阵必须是方阵，向量长度必须与矩阵维度匹配
    TORCH_CHECK(A.dim() == sip_pta::DIM_2 && A.size(0) == A.size(1),
                "asd_blas_ctrmv: Tensor A must be a 2D square matrix.");
    TORCH_CHECK(x.dim() == sip_pta::DIM_1 && x.size(0) == A.size(0),
                "asd_blas_ctrmv: Tensor x length must match matrix A dimension.");

    c10_npu::NPUGuard guard(A.device());
    // 提取张量维度和属性参数
    int64_t n = A.size(0);
    int64_t lda = n;
    int64_t incx = 1;

    // 转换枚举参数至底层类型
    AsdSip::asdBlasFillMode_t uplo_val = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
    AsdSip::asdBlasOperation_t trans_val = static_cast<AsdSip::asdBlasOperation_t>(trans);
    AsdSip::asdBlasDiagType_t diag_val = static_cast<AsdSip::asdBlasDiagType_t>(diag);

    // 构建传递给 getBlasHandle 的 Plan 参数列表
    std::vector<int64_t> planParam = {static_cast<int64_t>(uplo_val), n};

    // 定义 Plan 构建函数
    auto makePlan = [uplo_val, n](AsdSip::asdBlasHandle handle) {
        return AsdSip::asdBlasMakeCtrmvPlan(handle, uplo_val, n);
    };

    // 执行底层算子
    EXEC_BLAS_FUNC(AsdSip::asdBlasCtrmv, makePlan, planParam, uplo_val, trans_val, diag_val, n, A, lda, x, incx);

    // CTRMV 为原地修改算子，直接返回被修改后的 x
    return x;
}

// 算子注册模块
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    // x 为原地更新，使用 Tensor(a!) 标识可变引用
    m.def("asd_blas_ctrmv(Tensor A, Tensor(a!) x, int uplo, int trans, int diag) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_ctrmv", &asdBlasCtrmv); }
} // namespace sip_pta
