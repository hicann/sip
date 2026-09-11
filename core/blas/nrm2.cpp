/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "nrm2.h"
#include "blas_common.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
static constexpr uint32_t RESULT_SIZE = 1;
namespace AsdSip {
AspbStatus asdBlasNrm2Impl(asdBlasHandle handle, const int64_t n, aclTensor* x, int64_t incx, aclTensor* result,
                           int64_t scnrm)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Nrm2 get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t nrm2StorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &storageDims, &nrm2StorageDimsNum),
                                 "asdBlasNrm2Impl: aclGetStorageShape");
    int64_t xSize = scnrm == 1 ? *storageDims * ELEMENTS_EACH_COMPLEX64 : *storageDims;
    if (xSize != n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasNrm2Impl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(result, &storageDims, &nrm2StorageDimsNum),
                                 "asdBlasNrm2Impl: aclGetStorageShape");
    if (*storageDims != RESULT_SIZE) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasNrm2Impl get wrong result tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    delete[] storageDims;
    storageDims = nullptr;

    ASDSIP_ECHECK(n > 0, "blas asdBlasNrm2Impl get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasNrm2Impl get incx <= 0,change to 1.";
        incx = 1;
    }
    try {
        AsdSip::BlasPlan& plan = dynamic_cast<AsdSip::BlasPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Nrm2 plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "Nrm2Operation";
        AsdSip::OpParam::Nrm2 param;
        param.n = n;
        param.incx = incx;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> nrm2InTensors{x};
        SVector<aclTensor*> nrm2OutTensors{result};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, nrm2InTensors, nrm2OutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasNrm2Impl success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op nrm2 exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Nrm2 Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSnrm2(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSnrm2: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSnrm2 get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasSnrm2: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSnrm2 get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasSnrm2 get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    return asdBlasNrm2Impl(handle, n, x, incx, result, 0);
}

AspbStatus asdBlasScnrm2(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasScnrm2: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasScnrm2 get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasScnrm2: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasScnrm2 get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasScnrm2 get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    return asdBlasNrm2Impl(handle, ELEMENTS_EACH_COMPLEX64 * n, x, incx, result, 1);
}

AspbStatus asdBlasMakeNrm2Plan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeNrm2Plan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Nrm2 Plan failed: " << e.what();
        throw std::runtime_error("Make Nrm2 Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
