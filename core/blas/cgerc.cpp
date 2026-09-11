/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cgerc.h"
#include <complex>
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCgercPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct CgercTensorParam {
    const aclTensor* x;
    const aclTensor* y;
    const aclTensor* A;
};

AspbStatus CgercDtypeCheck(struct CgercTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.y, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgercShapeCheck(struct CgercTensorParam parm, const int64_t m, const int64_t n)
{
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= UINT32_MAX && n <= UINT32_MAX,
                  "blas Cgerc get m <= 0 || n <= 0 or m or n > 2^32.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, m, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.y, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgerc(asdBlasHandle handle, const int64_t m, const int64_t n, const std::complex<float>& alpha,
                        aclTensor* x, const int64_t incx, aclTensor* y, const int64_t incy, aclTensor* A,
                        const int64_t lda)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Cgerc get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CgercTensorParam parm {
        x, y, A
    };
    ASDSIP_CHECK(CgercDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCgerc dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_CHECK(CgercShapeCheck(parm, m, n) == ErrorType::ACL_SUCCESS, "blas asdBlasCgerc shape check failed.",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    if (incx != 1) {
        ASDSIP_LOG(INFO) << "blas asdBlasCgerc get incx != 1.";
    }
    if (incy != 1) {
        ASDSIP_LOG(INFO) << "blas asdBlasCgerc get incy != 1.";
    }
    if (lda <= 0) {
        ASDSIP_LOG(ERROR) << "blas asdBlasCgerc get lda <= 0.";
    }

    try {
        AsdSip::BlasCgercPlan& plan = dynamic_cast<AsdSip::BlasCgercPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cgerc plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "CgercOperation";
        AsdSip::OpParam::Cgerc param;
        param.m = m;
        param.n = n;
        param.incx = static_cast<uint32_t>(incx);
        param.incy = static_cast<uint32_t>(incy);
        param.alphaReal = alpha.real();
        param.alphaImag = alpha.imag();
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{x, y, plan.gatherAclOffset};
        SVector<aclTensor*> outTensors{A};
        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgerc success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op cgerc exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Cgerc Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCgercPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCgercPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCgercPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgercPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Cgerc Plan failed: " << e.what();
        throw std::runtime_error("Make Cgerc Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCgercPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
