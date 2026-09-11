/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "copy.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCopyPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int DOUBLE = 2;
namespace AsdSip {

struct CopyImplParam {
    int64_t n;
    aclTensor* x;
    int64_t incx;
    aclTensor* y;
    int64_t incy;
};

AspbStatus asdBlasCopyImpl(asdBlasHandle handle, CopyImplParam implParam, int64_t cCopy)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Copy get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t copyStorageDimsNum = 0;
    int64_t checkSize = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &copyStorageDimsNum),
                                 "asdBlasCopyImpl: aclGetStorageShape");
    checkSize = cCopy == 1 ? *storageDims * DOUBLE : *storageDims;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH) << "blas asdBlasCopyImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.y, &storageDims, &copyStorageDimsNum),
                                 "asdBlasCopyImpl: aclGetStorageShape");
    checkSize = cCopy == 1 ? *storageDims * DOUBLE : *storageDims;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH) << "blas asdBlasCopyImpl get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }
    ASDSIP_ECHECK(implParam.n > 0, "blas asdBlasCopyImpl get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);

    if (implParam.incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCopyImpl get incx <= 0,change incx =1.";
        implParam.incx = 1;
    }
    if (implParam.incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCopyImpl get incy <= 0,change incy =1.";
        implParam.incy = 1;
    }

    OpDesc opDesc;
    opDesc.opName = "CopyOperation";
    AsdSip::OpParam::Copy param;
    param.n = implParam.n;
    opDesc.specificParam = param;
    SVector<aclTensor*> inTensors{implParam.x};
    SVector<aclTensor*> outTensors{implParam.y};
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    try {
        AsdSip::BlasCopyPlan& plan = dynamic_cast<AsdSip::BlasCopyPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Copy plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCopy success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op copy exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Copy Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasScopy(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasScopy: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasScopy get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasScopy: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasScopy get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasScopy get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    CopyImplParam param = {n, x, incx, y, incy};
    return asdBlasCopyImpl(handle, param, 0);
}

AspbStatus asdBlasCcopy(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCcopy: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCcopy get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCcopy: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCcopy get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCcopy get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    CopyImplParam param = {DOUBLE * n, x, incx, y, incy};
    return asdBlasCopyImpl(handle, param, 1);
}

AspbStatus asdBlasMakeCopyPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCopyPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCopyPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCopyPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Copy Plan failed: " << e.what();
        throw std::runtime_error("Make Copy Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
