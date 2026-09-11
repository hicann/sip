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
 * @brief 复数矩阵-向量乘加算子 (CGEMV): y = alpha * op(A) * x + beta * y
 */
at::Tensor asdBlasCgemv(const at::Tensor& A, const at::Tensor& x, at::Tensor& y, const at::Scalar& alpha,
                        const at::Scalar& beta, int64_t trans)
{
    // 基本校验
    TORCH_CHECK(A.scalar_type() == at::kComplexFloat && x.scalar_type() == at::kComplexFloat &&
                    y.scalar_type() == at::kComplexFloat,
                "asd_blas_cgemv: Only ComplexFloat (Complex64) tensors are supported.");

    c10_npu::NPUGuard guard(A.device());
    // 参数准备
    const int64_t incx = 1;
    const int64_t incy = 1;
    AsdSip::asdBlasOperation_t opA = static_cast<AsdSip::asdBlasOperation_t>(trans);

    // 解析维度
    int64_t n = x.size(0);
    int64_t m = y.size(0);
    int64_t lda = A.size(0);

    // 转换标量 alpha, beta
    c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
    std::complex<float> a_std(a_c10.real(), a_c10.imag());

    c10::complex<float> b_c10 = beta.to<c10::complex<float>>();
    std::complex<float> b_std(b_c10.real(), b_c10.imag());

    auto acl_y = ConvertType(y);
    // 准备 Plan (注意处理 at::Tensor 到 aclTensor* 的转换)
    auto makePlan = [opA, m, n, acl_y, incy](AsdSip::asdBlasHandle handle) {
        return AsdSip::asdBlasMakeCgemvPlan(handle, opA, m, n, acl_y, incy);
    };
    std::vector<int64_t> params = {static_cast<int64_t>(opA), m, n, reinterpret_cast<int64_t>(acl_y), incy};
    // 执行算子
    // 传递 A, x, y (at::Tensor 类型)，宏内部的 ConvertTypes 会处理转换
    EXEC_BLAS_FUNC(AsdSip::asdBlasCgemv, makePlan, params, opA, m, n, a_std, A, lda, x, incx, b_std, acl_y, incy);

    return y;
}

// -----------------------------------------------------------------------------------
// 算子注册：不再包含 incx, incy
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_cgemv(Tensor A, Tensor x, Tensor(a!) y, Scalar alpha, Scalar beta, int trans) "
          "-> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cgemv", &asdBlasCgemv); }
} // namespace sip_pta
