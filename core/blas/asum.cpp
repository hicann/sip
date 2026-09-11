/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asum.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasAsumPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr uint32_t RESULT_SIZE = 1;

namespace AsdSip {

AspbStatus asdBlasSasumImpl(asdBlasHandle handle, const int64_t n, aclTensor* x, int64_t incx, aclTensor* result,
                            int64_t scaSum)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas asum get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    aclFormat tensorFormat = aclFormat::ACL_FORMAT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetFormat(x, &tensorFormat), "asdBlasSasumImpl: aclGetFormat");
    ASDSIP_ECHECK(tensorFormat == aclFormat::ACL_FORMAT_ND, "blas asum get invalid x tensor format.",
                  ErrorType::ACL_ERROR_FORMAT_NOT_MATCH);
    CHECK_STATUS_WITH_ACL_RETURN(aclGetFormat(result, &tensorFormat), "asdBlasSasumImpl: aclGetFormat");
    ASDSIP_ECHECK(tensorFormat == aclFormat::ACL_FORMAT_ND, "blas asum get invalid result tensor format.",
                  ErrorType::ACL_ERROR_FORMAT_NOT_MATCH);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasSasumImpl: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "get wrong out tensor type.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &storageDims, &storageDimsNum),
                                 "asdBlasSasumImpl: aclGetStorageShape");
    int64_t xSize = scaSum == 0 ? *storageDims : *storageDims * ELEMENTS_EACH_COMPLEX64;
    if (xSize != n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "get wrong in tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(result, &storageDims, &storageDimsNum),
                                 "asdBlasSasumImpl: aclGetStorageShape");
    if (*storageDims != RESULT_SIZE) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH) << "get wrong out tensor num.";
        return ErrorType::ACL_ERROR_OP_OUTPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asum get incx <= 0,change incx =1.";
        incx = 1;
    }

    try {
        AsdSip::BlasAsumPlan& plan = dynamic_cast<AsdSip::BlasAsumPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Asum plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "AsumOperation";
        AsdSip::OpParam::Asum param;
        param.n = static_cast<uint32_t>(n);
        param.incx = static_cast<uint32_t>(incx);
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> asumInTensors = {x};
        SVector<aclTensor*> asumOutTensors = {result};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, asumInTensors, asumOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasAsum success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op asum exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Asum Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSasum(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSasumImpl: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "get wrong in tensor type.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasScasum get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_LOG(INFO) << "Execute asdBlasSasum success.";
    return asdBlasSasumImpl(handle, n, x, incx, result, 0);
}

AspbStatus asdBlasScasum(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSasumImpl: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "get wrong in tensor type.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasScasum get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_LOG(INFO) << "Execute asdBlasScasum success.";
    return asdBlasSasumImpl(handle, ELEMENTS_EACH_COMPLEX64 * n, x, incx, result, 1);
}

AspbStatus asdBlasMakeAsumPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeAsumPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasAsumPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasAsumPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Asum Plan failed: " << e.what();
        throw std::runtime_error("Make Asum Plan failed.");
    }
    plan->MarkInitialized();
    ASDSIP_LOG(INFO) << "Execute asdBlasMakeAsumPlan success.";
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
