/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cgemv_batched.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCgemvBatchedPlan.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;
static constexpr uint32_t MAX_ELENUM_INLINE = 32; // max elements in line

namespace AsdSip {

struct CgemvBatchedImplParam {
    asdBlasOperation_t trans;
    asdDataType_t dtype;
    int64_t m;
    int64_t n;
    int64_t batchCount;
    aclTensor* A;
    aclTensor* x;
    aclTensor* y;
};

AspbStatus asdBlasCgemvBatchedShapeCheck(CgemvBatchedImplParam implParam)
{
    ASDSIP_ECHECK(implParam.m > 0 && implParam.n > 0 && implParam.m <= MAX_ELENUM_INLINE &&
                      implParam.n <= MAX_ELENUM_INLINE && implParam.batchCount > 0,
                  "blas asdBlasCgemvBatched get m <= 0 || n <= 0 || m > 32 || n > 32 || batchCount <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t* xstorageDims = nullptr;
    uint64_t xstorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &xstorageDims, &xstorageDimsNum),
                                 "asdBlasCgemvBatched: aclGetStorageShape");
    if (xstorageDimsNum > DIM_3 || *xstorageDims <= 0) {
        delete[] xstorageDims;
        xstorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasCgemvBatched get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (xstorageDims != nullptr) {
        delete[] xstorageDims;
        xstorageDims = nullptr;
    }

    int64_t* ystorageDims = nullptr;
    uint64_t ystorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.y, &ystorageDims, &ystorageDimsNum),
                                 "asdBlasCgemvBatched: aclGetStorageShape");
    if (ystorageDimsNum > DIM_3 || *ystorageDims <= 0) {
        delete[] ystorageDims;
        ystorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasCgemvBatched get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (ystorageDims != nullptr) {
        delete[] ystorageDims;
        ystorageDims = nullptr;
    }

    int64_t* AstorageDims = nullptr;
    uint64_t AstorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.A, &AstorageDims, &AstorageDimsNum),
                                 "asdBlasCgemvBatched: aclGetStorageShape");
    if (AstorageDimsNum != DIM_3 || AstorageDims[DIM_0] != implParam.batchCount || AstorageDims[DIM_1] != implParam.m ||
        AstorageDims[DIM_2] != implParam.n) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasCgemvBatched get wrong A tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (AstorageDims != nullptr) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgemvBatchedImpl(asdBlasHandle handle, CgemvBatchedImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas CgemvBatched get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AsdSip::AspbStatus shapeCheckStatus = asdBlasCgemvBatchedShapeCheck(implParam);
    if (shapeCheckStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return shapeCheckStatus;
    }

    try {
        AsdSip::BlasCgemvBatchedPlan& plan = dynamic_cast<AsdSip::BlasCgemvBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "CgemvBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;

        // y = alpha * A * x + beta * y, only support alpha = 1 & beta = 0
        OpDesc opDesc;
        opDesc.opName = "CgemvBatchedOperation";
        AsdSip::OpParam::CgemvBatched param;

        int64_t transType = static_cast<int64_t>(implParam.trans);
        int64_t dataType = static_cast<int64_t>(implParam.dtype);

        param = {transType, dataType, implParam.m, implParam.n, implParam.batchCount, plan.GetMaxMatNum()};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.A, implParam.x, plan.GetAclMaskTensor()};
        SVector<aclTensor*> outTensors{implParam.y};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgemvBatched success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "CgemvBatched Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasHCgemvBatched(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                                const std::complex<op::fp16_t>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                                const int64_t incx, const std::complex<op::fp16_t>& beta, aclTensor* y,
                                const int64_t incy, const int64_t batchCount)
{
    (void)alpha;
    (void)beta;
    // aclGetDataType does not support complex32 now, so no check datatype.
    std::lock_guard<std::mutex> lock(blas_mtx);
    if (lda <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasHCgemvBatched get lda <= 0.";
    }

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasHCgemvBatched get incx <= 0.";
    }

    if (incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasHCgemvBatched get incy <= 0.";
    }
    return asdBlasCgemvBatchedImpl(handle, {trans, asdDataType_t::ASD_C_32F, m, n, batchCount, A, x, y});
}

AspbStatus asdBlasCgemvBatched(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m, const int64_t n,
                               const std::complex<float>& alpha, aclTensor* A, const int64_t lda, aclTensor* x,
                               const int64_t incx, const std::complex<float>& beta, aclTensor* y, const int64_t incy,
                               const int64_t batchCount)
{
    (void)alpha;
    (void)beta;
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCgemvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCgemvBatched get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCgemvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCgemvBatched get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(A, &dataType), "asdBlasCgemvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCgemvBatched get wrong A tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    if (lda <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCgemvBatched get lda <= 0.";
    }

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCgemvBatched get incx <= 0.";
    }

    if (incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCgemvBatched get incy <= 0.";
    }
    return asdBlasCgemvBatchedImpl(handle, {trans, asdDataType_t::ASD_C_64F, m, n, batchCount, A, x, y});
}

AspbStatus asdBlasMakeCgemvBatchedPlanImpl(asdBlasHandle handle, asdBlasOperation_t trans, asdDataType_t dtype,
                                           const int64_t m)
{
    ASDSIP_ECHECK(handle != nullptr, "blas CgemvBatched Make Plan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(m > 0, "blas asdBlasMakeCgemvBatchedPlan get m <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCgemvBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgemvBatchedPlan(trans, dtype, m);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make CgemvBatched Plan failed: " << e.what();
        throw std::runtime_error("Make CgemvBatched Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas CgemvBatchedPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasMakeHCgemvBatchedPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    return asdBlasMakeCgemvBatchedPlanImpl(handle, trans, asdDataType_t::ASD_C_32F, m);
}

AspbStatus asdBlasMakeCgemvBatchedPlan(asdBlasHandle handle, asdBlasOperation_t trans, const int64_t m)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    return asdBlasMakeCgemvBatchedPlanImpl(handle, trans, asdDataType_t::ASD_C_64F, m);
}
} // namespace AsdSip
