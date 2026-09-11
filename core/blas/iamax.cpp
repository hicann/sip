/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "iamax.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasIamaxPlan.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t RESULT_SIZE = 1;

namespace AsdSip {

struct IamaxImplParam {
    int64_t n;
    aclTensor* x;
    int64_t incx;
    aclTensor* result;
    int64_t dtype;
};

AspbStatus asdBlasIamaxImpl(asdBlasHandle handle, IamaxImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Iamax get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t iamaxStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &iamaxStorageDimsNum),
                                 "asdBlasIamaxImpl: aclGetStorageShape");
    if (*storageDims != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasIamaxImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.result, &storageDims, &iamaxStorageDimsNum),
                                 "asdBlasIamaxImpl: aclGetStorageShape");
    if (*storageDims != RESULT_SIZE) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasIamaxImpl get wrong result tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    try {
        AsdSip::BlasIamaxPlan& plan = dynamic_cast<AsdSip::BlasIamaxPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Iamax plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        if (implParam.n <= 0 || implParam.incx <= 0) {
            implParam.result = plan.zeroAclTensor;
            return ErrorType::ACL_SUCCESS;
        }

        aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
        CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(implParam.x, &dataType), "asdBlasIamaxImpl: aclGetDataType");
        if ((dataType == aclDataType::ACL_FLOAT && implParam.dtype != 0) ||
            (dataType == aclDataType::ACL_COMPLEX64 && implParam.dtype != 1)) {
            ASDSIP_ELOG(ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE)
                << "blas asdBlasIamaxImpl x.dtype do not match dtype";
            return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
        }

        OpDesc opDesc;
        opDesc.opName = "IamaxOperation";
        AsdSip::OpParam::Iamax param;
        param.n = implParam.n;
        param.incx = implParam.incx;
        param.dtype = implParam.dtype;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.x};
        SVector<aclTensor*> outTensors{implParam.result};
        Status iamaxStatus = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(iamaxStatus.Ok(), iamaxStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasIamax success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op iamax exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Iamax Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasIsamax(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasIsamax: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasIsamax get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasIsamax: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_INT32, "blas asdBlasIsamax get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    return asdBlasIamaxImpl(handle, {n, x, incx, result, 0});
}

AspbStatus asdBlasIcamax(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasIcamax: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasIcamax get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(result, &dataType), "asdBlasIcamax: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_INT32, "blas asdBlasIcamax get wrong result tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    return asdBlasIamaxImpl(handle, {n, x, incx, result, 1});
}

AspbStatus asdBlasMakeIamaxPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeIamaxPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasIamaxPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasIamaxPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Iamax Plan failed: " << e.what();
        throw std::runtime_error("Make Iamax Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeIamaxPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
