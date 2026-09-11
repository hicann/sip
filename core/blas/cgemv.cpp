/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cgemv.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCgemvPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int TWO = 2;
constexpr int BASE_BLOCK_SIZE = 1024;
namespace AsdSip {

struct CgemvTensorParam {
    const aclTensor* A;
    const aclTensor* x;
    const aclTensor* y;
};

AspbStatus asdBlasCgemvParamCheck(int64_t lda, int64_t incx, int64_t incy)
{
    ASDSIP_ECHECK(lda > 0, "blas asdBlasCgemv get lda <= 0.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(incx == 1, "blas asdBlasCgemv get incx != 1.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(incy == 1, "blas asdBlasCgemv get incy != 1.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgemvDtypeCheck(struct CgemvTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.y, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgemvShapeCheck(struct CgemvTensorParam parm, const int64_t m, const int64_t n)
{
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= UINT32_MAX && n <= UINT32_MAX,
                  "blas asdBlasCgemv get m <= 0 || n <= 0 or m or n > 2^32.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.y, m, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgemv(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                        const std::complex<float>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                        const int64_t incx, const std::complex<float>& beta, aclTensor* y, const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas cgemv get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CgemvTensorParam parm {
        A, x, y
    };
    ASDSIP_CHECK(asdBlasCgemvParamCheck(lda, incx, incy) == ErrorType::ACL_SUCCESS,
                 "blas asdBlasCgemv param check failed.", return ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_CHECK(CgemvDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCgemv dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_CHECK(CgemvShapeCheck(parm, m, n) == ErrorType::ACL_SUCCESS, "blas asdBlasCgemv shape check failed.",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    try {
        AsdSip::BlasCgemvPlan& plan = dynamic_cast<AsdSip::BlasCgemvPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cgemv plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;

        // y = alpha * A * x + beta * y
        OpDesc opDesc;
        opDesc.opName = "CgemvOperation";
        AsdSip::OpParam::Cgemv param;

        ASDSIP_ECHECK(plan.GetyInAclTensor() != nullptr && plan.GetAclMaskTensor() != nullptr,
                      "blas Cgemv mask tensor is null.", ErrorType::ACL_ERROR_INVALID_PARAM);

        int64_t gTrans = -1;
        if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
            gTrans = 0;
        } else if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
            gTrans = 1;
        } else if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
            gTrans = TWO;
        }

        param = {gTrans, m, n, lda, incx, incy, alpha, beta};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{A, x, plan.GetyInAclTensor(), plan.GetAclMaskTensor()};
        SVector<aclTensor*> outTensors{y};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        y = outTensors.at(0);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgemv success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Cgemv Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCgemvPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                                aclTensor* y, const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas Cgemv Make Plan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= UINT32_MAX && n <= UINT32_MAX,
                  "blas asdBlasMakeCgemvPlan get m <= 0 || n <= 0 or m > 2^32 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(y != nullptr, "blas Cgemv y is nullptr.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(incy == 1, "blas Cgemv get incy != 1.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AsdSip::BlasCgemvPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgemvPlan(trans, m, n, y, incy);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Cgemv Plan failed: " << e.what();
        throw std::runtime_error("Make Cgemv Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCgemvPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
