/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ctrmv.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCtrmvPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int TWO = 2;

namespace AsdSip {
struct CtrmvTensorParam {
    const aclTensor* A;
    const aclTensor* x;
};

AspbStatus CtrmvDtypeCheck(struct CtrmvTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CtrmvShapeCheck(CtrmvTensorParam param, const int64_t n, const int64_t incx, const int64_t lda)
{
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCtrmv get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(incx > 0, "blas asdBlasCtrmv get incx < 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda > 0, "blas asdBlasCtrmv get lda <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    auto ret = ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    SIP_OP_CHECK_NUM_NOT_MATCH(param.A, n * n, ret);
    SIP_OP_CHECK_NUM_NOT_MATCH(param.x, n, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCtrmv(asdBlasHandle handle, asdBlasFillMode_t uplo, asdBlasOperation_t trans, asdBlasDiagType_t diag,
                        const int64_t n, aclTensor* A, const int64_t lda, aclTensor* x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Ctrmv get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CtrmvTensorParam parm {
        A, x
    };
    ASDSIP_CHECK(CtrmvDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCtrmv dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    auto ret = CtrmvShapeCheck(parm, n, incx, lda);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCtrmv shape check failed.", return ret);

    Status status;
    OpDesc opDesc;
    opDesc.opName = "CtrmvOperation";
    AsdSip::OpParam::Ctrmv param;

    int64_t transLocal = -1;
    if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
        transLocal = 0;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
        transLocal = 1;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
        transLocal = TWO;
    }

    int64_t uploLocal = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER) ? 0 : 1;
    int64_t diagLocal = (diag == asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT) ? 0 : 1;

    param.uplo = uploLocal;
    param.trans = transLocal;
    param.diag = diagLocal;
    param.n = n;
    param.lda = lda;
    param.incx = incx;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    try {
        AsdSip::BlasCtrmvPlan& plan = dynamic_cast<AsdSip::BlasCtrmvPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Ctrmv plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        SVector<aclTensor*> inTensors{A, x, plan.GetUploAclTensor()};
        SVector<aclTensor*> outTensors;
        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCtrmv success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the BlasCtrmv exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Ctrmv Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCtrmvPlan(asdBlasHandle handle, asdBlasFillMode_t uplo, int64_t n)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCtrmvPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCtrmv get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasCtrmvPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCtrmvPlan(uplo, n);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Ctrmv Plan failed: " << e.what();
        throw std::runtime_error("Make Ctrmv Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCtrmvPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
