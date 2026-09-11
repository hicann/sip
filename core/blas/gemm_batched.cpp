/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/opdev/common_types.h"
#include "aclnn/opdev/fp16_t.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasGemmBatchedPlan.h"
#include "hcgemm_batched.h"
#include "cgemm_batched.h"

using namespace AsdSip;
using namespace Mki;

constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int64_t PADNUM = 128;

namespace AsdSip {

static int64_t GetTrans(const asdBlasOperation_t trans)
{
    int64_t gTrans = -1;
    if (trans == asdBlasOperation_t::ASDBLAS_OP_N) {
        gTrans = 0;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
        gTrans = 1;
    } else if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
        gTrans = TWO;
    }
    return gTrans;
}

static AspbStatus checkStorage(aclTensor* tensor, const char* tensorName)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(tensor, &storageDims, &storageDimsNum),
                                 "asdBlasCgemm: aclGetStorageShape");
    if (*storageDims <= 0) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
            << "blas asdBlasCgemm get wrong " << tensorName << " tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    return ErrorType::ACL_SUCCESS;
}

static AspbStatus checkDtype(aclTensor& tensor, op::DataType dtype, const char* tensorName)
{
    op::DataType dataType = op::DataType::DT_UNDEFINED;
    dataType = tensor.GetDataType();

    std::string errMsg{"blas asdBlasCgemm get wrong "};
    errMsg = errMsg + tensorName + " tensor dtype.";
    ASDSIP_ECHECK(dataType == dtype, errMsg, ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasHCgemmBatched(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb,
                                const int64_t m, const int64_t n, const int64_t k,
                                const std::complex<op::fp16_t>& alpha, aclTensor* A, const int64_t lda, aclTensor* B,
                                const int64_t ldb, const std::complex<op::fp16_t>& beta, aclTensor* C,
                                const int64_t ldc, const int64_t batchCount)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas HCgemmBatched get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AspbStatus status;
    if ((status = checkStorage(A, "A")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkStorage(B, "B")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkStorage(C, "C")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*A, op::DataType::DT_COMPLEX32, "A")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*B, op::DataType::DT_COMPLEX32, "B")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*C, op::DataType::DT_COMPLEX32, "C")) != ErrorType::ACL_SUCCESS) {
        return status;
    }

    ASDSIP_ECHECK(m > 0 && k > 0 && n > 0, "blas CgemmBatched only supports m > 0 && k > 0 && n > 0.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(m <= 32 && k <= 32 && n <= 32, "blas CgemmBatched only supports m <= 32 && k <= 32 && n <= 32.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t gTransA = GetTrans(transa);
    ASDSIP_ECHECK(gTransA == 0, "blas HCgemmBatched only supports transa == 0.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    int64_t gTransB = GetTrans(transb);
    ASDSIP_ECHECK(gTransB == 0, "blas HCgemmBatched only supports transb == 0.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_ECHECK((alpha == std::complex<op::fp16_t>{1.0f, 0.0f}),
                  "blas HCgemmBatched only supports alpha == {1.0, 0.0}.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK((beta == std::complex<op::fp16_t>{0.0f, 0.0f}),
                  "blas HCgemmBatched only supports beta == {0.0, 0.0}.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_ECHECK(lda == k, "blas HCgemmBatched only supports lda == k.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(ldb == n, "blas HCgemmBatched only supports ldb == n.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(ldc == n, "blas HCgemmBatched only supports ldc == n.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    try {
        AsdSip::BlasHCgemmBatchedPlan& plan = dynamic_cast<AsdSip::BlasHCgemmBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "HCgemmBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "HCgemmBatchedOperation";
        AsdSip::OpParam::HCgemmBatched param;

        param = {m, k, n, batchCount};
        opDesc.specificParam = param;
        SVector<aclTensor*> inTensors;
        inTensors.push_back(A);
        inTensors.push_back(B);
        inTensors.push_back(plan.gatherOffsets);
        SVector<aclTensor*> outTensors;
        outTensors.push_back(C);
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        Status HCgemmBatchedStatus = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(HCgemmBatchedStatus.Ok(), HCgemmBatchedStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasHCgemmBatched success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op HCgemmBatched exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "HCgemmBatched Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeHCgemmBatchedPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeHCgemmBatchedPlan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasHCgemmBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasHCgemmBatchedPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make HCgemmBatched Plan failed: " << e.what();
        throw std::runtime_error("Make HCgemmBatched Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR)
            << "Fail to create blas asdBlasMakeHCgemmBatchedPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgemmBatched(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb,
                               const int64_t m, const int64_t n, const int64_t k, const std::complex<float>& alpha,
                               aclTensor* A, const int64_t lda, aclTensor* B, const int64_t ldb,
                               const std::complex<float>& beta, aclTensor* C, const int64_t ldc,
                               const int64_t batchCount)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas CgemmBatched get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    AspbStatus status;
    if ((status = checkStorage(A, "A")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkStorage(B, "B")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkStorage(C, "C")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*A, op::DataType::DT_COMPLEX64, "A")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*B, op::DataType::DT_COMPLEX64, "B")) != ErrorType::ACL_SUCCESS) {
        return status;
    }
    if ((status = checkDtype(*C, op::DataType::DT_COMPLEX64, "C")) != ErrorType::ACL_SUCCESS) {
        return status;
    }

    ASDSIP_ECHECK(m > 0 && k > 0 && n > 0, "blas CgemmBatched only supports m > 0 && k > 0 && n > 0.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(m <= 32 && k <= 32 && n <= 32, "blas CgemmBatched only supports m <= 32 && k <= 32 && n <= 32.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t gTransA = GetTrans(transa);
    ASDSIP_ECHECK(gTransA == 0, "blas CgemmBatched only supports transa == 0.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    int64_t gTransB = GetTrans(transb);
    ASDSIP_ECHECK(gTransB == 0, "blas CgemmBatched only supports transb == 0.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_ECHECK((alpha == std::complex<float>{1.0f, 0.0f}), "blas CgemmBatched only supports alpha == {1.0, 0.0}.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK((beta == std::complex<float>{0.0f, 0.0f}), "blas CgemmBatched only supports beta == {0.0, 0.0}.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_ECHECK(lda == k, "blas CgemmBatched only supports lda == k.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(ldb == n, "blas CgemmBatched only supports ldb == n.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(ldc == n, "blas CgemmBatched only supports ldc == n.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    try {
        AsdSip::BlasCgemmBatchedPlan& plan = dynamic_cast<AsdSip::BlasCgemmBatchedPlan&>(
            BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "CgemmBatched plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "CgemmBatchedOperation";
        AsdSip::OpParam::CgemmBatched param;

        param = {m, k, n, batchCount};
        opDesc.specificParam = param;
        SVector<aclTensor*> inTensors;
        inTensors.push_back(A);
        inTensors.push_back(B);
        inTensors.push_back(plan.gatherOffsets);
        SVector<aclTensor*> outTensors;
        outTensors.push_back(C);
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        Status CgemmBatchedStatus = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(CgemmBatchedStatus.Ok(), CgemmBatchedStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgemmBatched success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op CgemmBatched exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "CgemmBatched Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCgemmBatchedPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCgemmBatchedPlan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCgemmBatchedPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgemmBatchedPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make CgemmBatched Plan failed: " << e.what();
        throw std::runtime_error("Make CgemmBatched Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR)
            << "Fail to create blas asdBlasMakeCgemmBatchedPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
