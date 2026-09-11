/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "caxpy.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCaxpyPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct CaxpTensorParam {
    const aclTensor* x;
    const aclTensor* y;
};

AspbStatus CaxpDtypeCheck(struct CaxpTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.y, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CaxpShapeCheck(struct CaxpTensorParam parm, const int64_t n)
{
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCaxpy get n <= 0 or n > 2^32",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.x, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.y, n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCaxpy(asdBlasHandle handle, const int64_t n, const std::complex<float>& alpha, aclTensor* x,
                        int64_t incx, aclTensor* y, int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Caxpy get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CaxpTensorParam parm {
        x, y
    };
    ASDSIP_CHECK(CaxpDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasCaxpy dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_CHECK(CaxpShapeCheck(parm, n) == ErrorType::ACL_SUCCESS, "blas asdBlasCaxpy shape check failed.",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCaxpy get incx <= 0,change incx =1.";
        incx = 1;
    }

    if (incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCaxpy get incy <= 0,change incy =1.";
        incy = 1;
    }

    try {
        AsdSip::BlasCaxpyPlan& plan = dynamic_cast<AsdSip::BlasCaxpyPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cal plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status caxpyStatus;
        OpDesc opDesc;
        opDesc.opName = "CaxpyOperation";
        AsdSip::OpParam::Caxpy param;
        param.n = static_cast<uint32_t>(n);
        param.incx = static_cast<uint32_t>(incx);
        param.incy = static_cast<uint32_t>(incy);
        param.alphaReal = alpha.real();
        param.alphaImag = alpha.imag();
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors = {x, plan.augAclTensor};
        SVector<aclTensor*> outTensors = {y};

        caxpyStatus = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(caxpyStatus.Ok(), caxpyStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCaxpy success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op Caxpy exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Caxpy Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCaxpyPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas asdBlasMakeCaxpyPlan Failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasCaxpyPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCaxpyPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Caxpy Plan failed: " << e.what();
        throw std::runtime_error("Make Caxpy Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCaxpyPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
