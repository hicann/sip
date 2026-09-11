/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "colwise_mul.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasColwiseMulPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct ColwiseMuParam {
    int64_t m;
    int64_t n;
    aclTensor* mat;
    aclTensor* vec;
    aclTensor* result;
};

AspbStatus ColwiseMulDtypeCheck(struct ColwiseMuParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.mat, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.vec, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.result, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus ColwiseMulShapeCheck(struct ColwiseMuParam parm)
{
    int64_t m = parm.m;
    int64_t n = parm.n;
    ASDSIP_ECHECK(m > 0 && n > 0 && m <= UINT32_MAX && n <= UINT32_MAX,
                  "blas ColwiseMul get m <= 0 || n <= 0 or m or n > 2^32.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.mat, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.vec, m, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.result, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasColwiseMul(asdBlasHandle handle, const int64_t m, const int64_t n, aclTensor* mat, aclTensor* vec,
                             aclTensor* result)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas ColwiseMul get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ColwiseMuParam implParam = {m, n, mat, vec, result};
    ASDSIP_CHECK(ColwiseMulDtypeCheck(implParam) == ErrorType::ACL_SUCCESS,
                 "blas asdBlasColwiseMul dtype check failed.", return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    ASDSIP_CHECK(ColwiseMulShapeCheck(implParam) == ErrorType::ACL_SUCCESS,
                 "blas asdBlasColwiseMul shape check failed.", return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    try {
        AsdSip::BlasColwiseMulPlan& plan = dynamic_cast<AsdSip::BlasColwiseMulPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "ColwiseMul plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;
        OpDesc opDesc;
        opDesc.opName = "ColwiseMulOperation";
        AsdSip::OpParam::ColwiseMul param;
        param.m = m;
        param.n = n;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> colwiseMulInTensors{mat, vec, plan.augAclTensor};
        SVector<aclTensor*> colwiseMulOutTensors{result};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, colwiseMulInTensors, colwiseMulOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasColwiseMul success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the BlasColwiseMul exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "ColwiseMul Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeColwiseMulPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeColwiseMulPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    AsdSip::BlasColwiseMulPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasColwiseMulPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make ColwiseMul Plan failed: " << e.what();
        throw std::runtime_error("Make ColwiseMul Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR)
            << "Fail to create blas asdBlasMakeColwiseMulPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
