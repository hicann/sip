/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cal.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCalPlan.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr uint32_t BUFFER_SIZE16 = 16 * 1024;
namespace AsdSip {

struct CalImplParam {
    int64_t n;
    float alphaReal;
    float alphaImag;
    aclTensor* x;
    int64_t incx;
};

AspbStatus asdBlasCalImpl(OpParam::Cal::CalType calType, asdBlasHandle handle, CalImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Cal get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &storageDimsNum),
                                 "asdBlasCalImpl: aclGetStorageShape");
    int64_t xSize = calType == OpParam::Cal::CalType::CAL_CSSCAL ? *storageDims * ELEMENTS_EACH_COMPLEX64 :
                                                                   *storageDims;
    if (xSize != implParam.n) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "Input size mismatch between x and implParam.n.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    if (implParam.incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCalImpl get incx <= 0,change incx =1.";
        implParam.incx = 1;
    }

    Status status;
    OpDesc opDesc;
    opDesc.opName = "CalOperation";
    AsdSip::OpParam::Cal param;
    param.n = implParam.n;
    param.alphaReal = implParam.alphaReal;
    param.alphaImag = implParam.alphaImag;
    param.calType = calType;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    try {
        AsdSip::BlasCalPlan& plan = dynamic_cast<AsdSip::BlasCalPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cal plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        aclTensor* mask = plan.GetAclMaskTensor();
        ASDSIP_ECHECK(mask != nullptr, "blas cal mask tensor is null!", ErrorType::ACL_ERROR_INTERNAL_ERROR);
        SVector<aclTensor*> inTensors = {implParam.x};
        SVector<aclTensor*> outTensors;
        if (calType == OpParam::Cal::CalType::CAL_CSCAL) {
            inTensors.push_back(mask);
        }

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCal success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the BlasCal exception ..
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "BlasCal Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasSscal(asdBlasHandle handle, const int64_t n, const float& alpha, aclTensor* x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasSscal: aclGetDataType");

    ASDSIP_ECHECK(dataType == aclDataType::ACL_FLOAT, "blas asdBlasSscal get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasSscal get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    float alphaReal = alpha;
    float alphaImag = 0.0;
    OpParam::Cal::CalType calType = OpParam::Cal::CalType::CAL_SSCAL;

    return asdBlasCalImpl(calType, handle, {n, alphaReal, alphaImag, x, incx});
}

AspbStatus asdBlasCsscal(asdBlasHandle handle, const int64_t n, const float& alpha, aclTensor* x, const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCsscal: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCsscal get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCsscal get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    float alphaReal = alpha;
    float alphaImag = 0.0;
    OpParam::Cal::CalType calType = OpParam::Cal::CalType::CAL_CSSCAL;
    return asdBlasCalImpl(calType, handle, {ELEMENTS_EACH_COMPLEX64 * n, alphaReal, alphaImag, x, incx});
}

AspbStatus asdBlasCscal(asdBlasHandle handle, const int64_t n, const std::complex<float>& alpha, aclTensor* x,
                        const int64_t incx)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(x != nullptr, "blas cscal tensor x is null!", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCscal: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCscal get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCscal get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    float alphaReal = alpha.real();
    float alphaImag = alpha.imag();
    OpParam::Cal::CalType calType = OpParam::Cal::CalType::CAL_CSCAL;
    return asdBlasCalImpl(calType, handle, {n, alphaReal, alphaImag, x, incx});
}

AspbStatus asdBlasMakeCalPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas asdBlasMakeCalPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasCalPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCalPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Cal Plan failed: " << e.what();
        throw std::runtime_error("Make Cal Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasCal mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
