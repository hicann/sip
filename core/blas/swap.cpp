/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "swap.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasSwapPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {

constexpr int64_t complexNum = 2;

struct SwapImplParam {
    int64_t n;
    aclTensor* x;
    int64_t incx;
    aclTensor* y;
    int64_t incy;
};

AspbStatus asdBlasSwapImpl(asdBlasHandle handle, SwapImplParam implParam, int64_t cswap)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Swap get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t swapStorageDimsNum = 0;
    int64_t checkSize = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &swapStorageDimsNum),
                                 "asdBlasSwapImpl: aclGetStorageShape");
    checkSize = cswap == 1 ? *storageDims * complexNum : *storageDims;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasSwapImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.y, &storageDims, &swapStorageDimsNum),
                                 "asdBlasSwapImpl: aclGetStorageShape");
    checkSize = cswap == 1 ? *storageDims * complexNum : *storageDims;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasSwapImpl get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    if (implParam.incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSwapImpl get incx <= 0.";
        implParam.incx = 1;
    }

    if (implParam.incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSwapImpl get incy <= 0.";
        implParam.incy = 1;
    }

    Status status;
    OpDesc opDesc;
    opDesc.opName = "SwapOperation";
    AsdSip::OpParam::Swap param;
    param.n = implParam.n;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> swapInTensors{implParam.x, implParam.y};
    SVector<aclTensor*> swapOutTensors{};

    try {
        AsdSip::BlasSwapPlan& plan = dynamic_cast<AsdSip::BlasSwapPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Swap plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        status = RunAsdOpsV2(plan.GetStream(), opDesc, swapInTensors, swapOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasSwap success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op Swap exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Swap Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSswap(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSswap: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSswap get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasSswap: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSswap get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasSswap get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    return asdBlasSwapImpl(handle, {n, x, incx, y, incy}, 0);
}

AspbStatus asdBlasCswap(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCswap: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCswap get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCswap: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCswap get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasSswap get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    return asdBlasSwapImpl(handle, {complexNum * n, x, incx, y, incy}, 1);
}

AspbStatus asdBlasMakeSwapPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeSwapPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasSwapPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasSwapPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Swap Plan failed: " << e.what();
        throw std::runtime_error("Make Swap Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
