/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cmatinv_batched.h"
#include "cgetri_batched.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCmatinvBatchedPlan.h"
#include "blasplan/include/blasplan/BlasCgetriBatchedPlan.h"

using namespace AsdSip;
using namespace Mki;

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_COMPLEX32 = 0;
static constexpr int64_t DTYPE_COMPLEX64 = 1;

namespace AsdSip {

struct CmatinvBatchedImplParam {
    int64_t dtype;
    int64_t n;
    int64_t batchSize;
    aclTensor* A;
    aclTensor* Ainv;
};

AspbStatus asdBlasCmatinvBatchedShapeCheck(CmatinvBatchedImplParam implParam)
{
    int64_t* AstorageDims = nullptr;
    uint64_t AstorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.A, &AstorageDims, &AstorageDimsNum),
                                 "asdBlasCmatinvBatched: aclGetStorageShape");
    if (AstorageDimsNum != DIM_3 || AstorageDims[DIM_0] != implParam.batchSize || AstorageDims[DIM_1] != implParam.n ||
        AstorageDims[DIM_2] != implParam.n) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
            << "blas asdBlasCmatinvBatched get wrong A tensor dim num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (AstorageDims != nullptr) {
        delete[] AstorageDims;
        AstorageDims = nullptr;
    }

    int64_t* AinvStorageDims = nullptr;
    uint64_t AinvStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.Ainv, &AinvStorageDims, &AinvStorageDimsNum),
                                 "asdBlasCmatinvBatched: aclGetStorageShape");
    if (AinvStorageDimsNum != DIM_3 || AinvStorageDims[DIM_0] != implParam.batchSize ||
        AinvStorageDims[DIM_1] != implParam.n || AinvStorageDims[DIM_2] != implParam.n) {
        delete[] AinvStorageDims;
        AinvStorageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
            << "blas asdBlasCmatinvBatched get wrong Ainv tensor dim num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (AinvStorageDims != nullptr) {
        delete[] AinvStorageDims;
        AinvStorageDims = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

// n < 32
AspbStatus runCmatinvBatchedOps(asdBlasHandle handle, CmatinvBatchedImplParam implParam)
{
    try {
        AsdSip::BlasCmatinvBatchedPlan& plan = dynamic_cast<AsdSip::BlasCmatinvBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "CmatinvBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "CmatinvBatchedOperation";
        AsdSip::OpParam::CmatinvBatched param;

        param = {implParam.dtype, implParam.n, implParam.batchSize};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.A, plan.uniMatAclReal, plan.uniMatAclImag, plan.offsetAclTensor};
        SVector<aclTensor*> outTensors{implParam.Ainv};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCmatinvBatched success when n < 32.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "CmatinvBatched Error (n < 32): " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

// n >= 32
AspbStatus runCgetriBatchedOps(asdBlasHandle handle, CmatinvBatchedImplParam implParam)
{
    try {
        AsdSip::BlasCgetriBatchedPlan& plan = dynamic_cast<AsdSip::BlasCgetriBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "CmatinvBatched (n >= 32) plan init Error!.",
                      ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "CgetriBatchedOperation";
        AsdSip::OpParam::CgetriBatched param;

        param = {implParam.dtype, implParam.n, implParam.batchSize};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.A,           plan.wBatchAcl,        plan.gather1OffsetAcl,
                                      plan.gather2OffsetAcl, plan.gather3OffsetAcl, plan.eyeBatchMatAcl,
                                      plan.aWorkAcl,         plan.workGmAcl};
        SVector<aclTensor*> outTensors{implParam.Ainv};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCmatinvBatched success when n >= 32.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "CmatinvBatched Error (n >= 32): " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasCmatinvBatchedImpl(asdBlasHandle handle, CmatinvBatchedImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas CmatinvBatched get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AsdSip::AspbStatus shapeCheckStatus = asdBlasCmatinvBatchedShapeCheck(implParam);
    if (shapeCheckStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return shapeCheckStatus;
    }

    AsdSip::AspbStatus runOpsStatus;
    if (implParam.n < MATRIX_SHAPE_LIMIT) {
        runOpsStatus = runCmatinvBatchedOps(handle, implParam);
    } else {
        runOpsStatus = runCgetriBatchedOps(handle, implParam);
    }
    return runOpsStatus;
}

AspbStatus asdBlasCmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                 aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize)
{
    (void)info;
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(A, &dataType), "asdBlasCmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCmatinvBatched get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(Ainv, &dataType), "asdBlasCmatinvBatched: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCmatinvBatched get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    // 对齐 ops-solver 行为与文档约束「当前约束为 n」：lda/lda_inv 非法时显式拦截（issue #138）
    ASDSIP_ECHECK(lda > 0 && lda_inv > 0, "blas asdBlasCmatinvBatched get lda <= 0 or lda_inv <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n && lda_inv == n, "blas asdBlasCmatinvBatched only supports lda == n and lda_inv == n.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n > 0 && batchSize > 0, "blas asdBlasCmatinvBatched get n <= 0 || batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "blas asdBlasCmatinvBatched get n <= 256 || batchSize <= 3000.", ErrorType::ACL_ERROR_INVALID_PARAM);

    return asdBlasCmatinvBatchedImpl(handle, {DTYPE_COMPLEX64, n, batchSize, A, Ainv});
}

AspbStatus asdBlasHCmatinvBatched(asdBlasHandle handle, const int64_t n, aclTensor* A, const int64_t lda,
                                  aclTensor* Ainv, const int64_t lda_inv, aclTensor* info, int64_t batchSize)
{
    (void)info;
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(n > 0 && batchSize > 0, "blas asdBlasHCmatinvBatched get n <= 0 || batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    ASDSIP_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "blas asdBlasHCmatinvBatched get n <= 256 || batchSize <= 3000.", ErrorType::ACL_ERROR_INVALID_PARAM);

    // 对齐 ops-solver 行为与文档约束「当前约束为 n」：lda/lda_inv 非法时显式拦截（issue #138）
    ASDSIP_ECHECK(lda > 0 && lda_inv > 0, "blas asdBlasHCmatinvBatched get lda <= 0 or lda_inv <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(lda == n && lda_inv == n, "blas asdBlasHCmatinvBatched only supports lda == n and lda_inv == n.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    return asdBlasCmatinvBatchedImpl(handle, {DTYPE_COMPLEX32, n, batchSize, A, Ainv});
}

AspbStatus makeCmatinvBatchedPlan(asdBlasHandle handle, const int64_t n)
{
    AsdSip::BlasCmatinvBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCmatinvBatchedPlan(n);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make CmatinvBatched Plan failed: " << e.what();
        throw std::runtime_error("Make CmatinvBatched Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas CmatinvBatched mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}

AspbStatus makeCgetriBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize, int64_t dtype)
{
    AsdSip::BlasCgetriBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgetriBatchedPlan(n, batchSize, dtype);
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make CmatinvBatched (n >= 32) Plan failed: " << e.what();
        throw std::runtime_error("Make CmatinvBatched (n >= 32) Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas CmatinvBatched (n >= 32) mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasMakeCmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas CmatinvBatched Make Plan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n > 0, "blas asdBlasMakeCmatinvBatchedPlan get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(batchSize > 0, "blas asdBlasMakeCmatinvBatchedPlan get batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    if (n < MATRIX_SHAPE_LIMIT) {
        return makeCmatinvBatchedPlan(handle, n);
    } else {
        return makeCgetriBatchedPlan(handle, n, batchSize, DTYPE_COMPLEX64);
    }
}

AspbStatus asdBlasMakeHCmatinvBatchedPlan(asdBlasHandle handle, const int64_t n, const int64_t batchSize)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas HCmatinvBatched Make Plan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n > 0, "blas asdBlasMakeHCmatinvBatchedPlan get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(batchSize > 0, "blas asdBlasMakeHCmatinvBatchedPlan get batchSize <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    if (n < MATRIX_SHAPE_LIMIT) {
        return makeCmatinvBatchedPlan(handle, n);
    } else {
        return makeCgetriBatchedPlan(handle, n, batchSize, DTYPE_COMPLEX32);
    }
}
} // namespace AsdSip
