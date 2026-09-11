/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dot.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasDotPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int ELEMENTS_EACH_COMPLEX64 = 2;

namespace AsdSip {

struct DotImplParam {
    int64_t n;
    aclTensor* x;
    int64_t incx;
    aclTensor* y;
    int64_t incy;
    aclTensor* result;
    int64_t isConj;
};

AspbStatus asdBlasDotImpl(asdBlasHandle handle, DotImplParam implParam, int64_t sdot)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Dot get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t dotStorageDimsNum = 0;
    int64_t checkSize = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &dotStorageDimsNum),
                                 "asdBlasDotImpl: aclGetStorageShape");
    checkSize = sdot == 1 ? *storageDims : *storageDims * ELEMENTS_EACH_COMPLEX64;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasDotImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.y, &storageDims, &dotStorageDimsNum),
                                 "asdBlasDotImpl: aclGetStorageShape");
    checkSize = sdot == 1 ? *storageDims : *storageDims * ELEMENTS_EACH_COMPLEX64;
    if (checkSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasDotImpl get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    ASDSIP_ECHECK(implParam.n > 0, "blas asdBlasDotImpl get wrong input n, please check.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    if (implParam.incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasDotImpl get incx <= 0,change to 1.";
        implParam.incx = 1;
    }

    if (implParam.incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasDotImpl get incy <= 0,change to 1.";
        implParam.incy = 1;
    }

    Status status;
    OpDesc opDesc;
    opDesc.opName = "DotOperation";
    AsdSip::OpParam::Dot param;
    param.n = implParam.n;
    param.isConj = implParam.isConj;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> dotInTensors{implParam.x, implParam.y};
    SVector<aclTensor*> dotOutTensors{implParam.result};

    try {
        AsdSip::BlasDotPlan& plan = dynamic_cast<AsdSip::BlasDotPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Dot plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        status = RunAsdOpsV2(plan.GetStream(), opDesc, dotInTensors, dotOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasDot success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op dot exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Dot Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSdot(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                       const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasSdot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSdot get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    DotImplParam param = {n, x, incx, y, incy, result, 0};
    return asdBlasDotImpl(handle, param, 1);
}

AspbStatus asdBlasCdotu(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasCdotu: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotu get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCdotu get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    DotImplParam param = {ELEMENTS_EACH_COMPLEX64 * n, x, incx, y, incy, result, 0};
    return asdBlasDotImpl(handle, param, 0);
}

AspbStatus asdBlasCdotc(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasCdotc: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCdotc get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCdotc get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    DotImplParam param = {ELEMENTS_EACH_COMPLEX64 * n, x, incx, y, incy, result, 1};
    return asdBlasDotImpl(handle, param, 0);
}

AspbStatus asdBlasMakeDotPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeDotPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasDotPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasDotPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Dot Plan failed: " << e.what();
        throw std::runtime_error("Make Dot Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
