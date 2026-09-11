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
 * @brief 复数矩阵-向量批量乘加算子 (CgemvBatched): y = alpha * op(A) * x + beta * y (Batch 模式)
 */
at::Tensor asdBlasCgemvBatched(const at::Tensor& A, const at::Tensor& x, at::Tensor& y, const at::Scalar& alpha,
                               const at::Scalar& beta, int64_t trans)
{
    // 1. 基本校验：支持 ComplexFloat 和 ComplexHalf
    TORCH_CHECK((A.scalar_type() == at::kComplexFloat || A.scalar_type() == at::kComplexHalf) &&
                    A.scalar_type() == x.scalar_type() && A.scalar_type() == y.scalar_type(),
                "asd_blas_cgemv_batched: Tensors must be ComplexFloat or ComplexHalf and have "
                "matching types.");

    c10_npu::NPUGuard guard(A.device());

    // 2. 参数准备
    const int64_t incx = 1;
    const int64_t incy = 1;
    AsdSip::asdBlasOperation_t opA = static_cast<AsdSip::asdBlasOperation_t>(trans);
    int64_t batchCount = A.size(0);
    int64_t m_val = y.size(1);
    int64_t n_val = x.size(1);
    int64_t lda = 0; // 该算子下为无效参数

    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    std::vector<int64_t> params = {static_cast<int64_t>(opA), m_val};

    // 3. 根据数据类型分发调用
    if (A.scalar_type() == at::kComplexFloat) {
        // 单精度逻辑 (Complex64)
        c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
        std::complex<float> a_std(a_c10.real(), a_c10.imag());
        c10::complex<float> b_c10 = beta.to<c10::complex<float>>();
        std::complex<float> b_std(b_c10.real(), b_c10.imag());

        auto makePlan = [opA, m_val](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeCgemvBatchedPlan(handle, opA, m_val);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasCgemvBatched, makePlan, params, opA, m_val, n_val, a_std, acl_a, lda, x, incx,
                       b_std, y, incy, batchCount);
    } else {
        // 半精度逻辑 (Complex32)
        // 注意：此处需要将 Scalar 转换为半精度复数，op::fp16_t 是底层定义
        c10::complex<at::Half> a_c10 = alpha.to<c10::complex<at::Half>>();
        std::complex<op::fp16_t> a_std(a_c10.real().x, a_c10.imag().x);

        c10::complex<at::Half> b_c10 = beta.to<c10::complex<at::Half>>();
        std::complex<op::fp16_t> b_std(b_c10.real().x, b_c10.imag().x);

        auto makePlan = [opA, m_val](AsdSip::asdBlasHandle handle) {
            return AsdSip::asdBlasMakeHCgemvBatchedPlan(handle, opA, m_val);
        };

        EXEC_BLAS_FUNC(AsdSip::asdBlasHCgemvBatched, makePlan, params, opA, m_val, n_val, a_std, acl_a, lda, x, incx,
                       b_std, y, incy, batchCount);
    }

    return y;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_cgemv_batched(Tensor A, Tensor x, Tensor(a!) y, Scalar alpha, Scalar beta, int "
          "trans) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cgemv_batched", &asdBlasCgemvBatched); }
} // namespace sip_pta
