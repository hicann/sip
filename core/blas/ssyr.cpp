/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ssyr.h"
#include "blas_common.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
// A = alpha * x * x.T + A
AspbStatus asdBlasSsyr(asdBlasHandle handle, asdBlasFillMode_t uplo, const int64_t n, const float& alpha, aclTensor* x,
                       const int64_t incx, aclTensor* A, const int64_t lda)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Ssyr get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t ssyrStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(x, &storageDims, &ssyrStorageDimsNum),
                                 "asdBlasSsyr: aclGetStorageShape");
    if (*storageDims != n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasSsyr get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    delete[] storageDims;
    storageDims = nullptr;

    // 对齐 ssyr2 的校验模式：A 必须为 (n, n)，防止元素数不足 n*n 时 kernel 越界写（issue #130）
    SIP_OP_CHECK_SHAPE_NOT_MATCH(A, (std::vector<int64_t>{n, n}), ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSsyr: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSsyr get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(A, &dataType), "asdBlasSsyr: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSsyr get wrong A tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(n > 0, "blas asdBlasSsyr get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSsyr get incx <= 0.";
    }

    if (lda <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSsyr get lda <= 0.";
    }

    try {
        AsdSip::BlasPlan& plan = dynamic_cast<AsdSip::BlasPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Ssyr plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "SsyrOperation";
        AsdSip::OpParam::Ssyr param;

        uint32_t uploFlag = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER) ? 0 : 1;
        param = {uploFlag, static_cast<uint32_t>(n), static_cast<uint32_t>(incx), alpha};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> ssyrInTensors{x};
        SVector<aclTensor*> ssyrOutTensors{A};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, ssyrInTensors, ssyrOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasSsyr success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op ssyr exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Ssyr Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeSsyrPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeSsyrPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
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
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Ssyr Plan failed: " << e.what();
        throw std::runtime_error("Make Ssyr Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeSsyrPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
