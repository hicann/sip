/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cgemm.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasCgemmPlan.h"

using namespace AsdSip;
using namespace Mki;

constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int64_t PADNUM = 128;

namespace AsdSip {
int64_t GetTrans(const asdBlasOperation_t trans)
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

struct CgemmTensorParam {
    const aclTensor* A;
    const aclTensor* B;
    const aclTensor* C;
};

AspbStatus asdBlasCgemmParamCheck(int64_t lda, int64_t ldb, int64_t ldc)
{
    ASDSIP_ECHECK(lda > 0, "blas asdBlasCgemm get lda <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(ldb > 0, "blas asdBlasCgemm get ldb <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(ldc > 0, "blas asdBlasCgemm get ldc <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgemmDtypeCheck(struct CgemmTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.B, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.C, aclDataType::ACL_COMPLEX64, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CgemmShapeCheck(struct CgemmTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    SIP_OP_CHECK_INVALID_SHAPE(parm.A, ret);
    SIP_OP_CHECK_INVALID_SHAPE(parm.B, ret);
    SIP_OP_CHECK_INVALID_SHAPE(parm.C, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasCgemm(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb, const int64_t m,
                        const int64_t n, const int64_t k, const std::complex<float>& alpha, aclTensor* A,
                        const int64_t lda, aclTensor* B, const int64_t ldb, const std::complex<float>& beta,
                        aclTensor* C, const int64_t ldc)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Cgemm get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct CgemmTensorParam parm {
        A, B, C
    };
    auto ret = CgemmDtypeCheck(parm);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCgemm dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ret = CgemmShapeCheck(parm);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCgemm shape check failed.",
                 return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    ret = asdBlasCgemmParamCheck(lda, ldb, ldc);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasCgemm param check failed.",
                 return ErrorType::ACL_ERROR_INVALID_PARAM);

    try {
        AsdSip::BlasCgemmPlan& plan = dynamic_cast<AsdSip::BlasCgemmPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Cgemm plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "CgemmOperation";
        AsdSip::OpParam::Cgemm param;

        int64_t gTransA = GetTrans(transa);
        int64_t gTransB = GetTrans(transb);
        int64_t ldaPad = (transa == asdBlasOperation_t::ASDBLAS_OP_N) ? ((m + PADNUM - 1) / PADNUM * PADNUM) :
                                                                        ((k + PADNUM - 1) / PADNUM * PADNUM);
        int64_t ldbPad = (transb == asdBlasOperation_t::ASDBLAS_OP_N) ? ((k + PADNUM - 1) / PADNUM * PADNUM) :
                                                                        ((n + PADNUM - 1) / PADNUM * PADNUM);
        param = {m, n, k, gTransA, gTransB, lda, ldb, ldc, ldaPad, ldbPad, alpha, beta};
        opDesc.specificParam = param;

        if (plan.augAAclTensors.size() != 2 || plan.augBAclTensors.size() != 2 || plan.augCAclTensors.size() != 4) {
            ASDSIP_LOG(ERROR) << "mask tensors error !";
            return ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        SVector<aclTensor*> inTensors = {A, B};
        for (auto tensor : plan.augAAclTensors) {
            inTensors.push_back(tensor);
        }
        for (auto tensor : plan.augBAclTensors) {
            inTensors.push_back(tensor);
        }
        for (auto tensor : plan.augCAclTensors) {
            inTensors.push_back(tensor);
        }

        SVector<aclTensor*> outTensors = {C};
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        Status cgemmStatus = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(cgemmStatus.Ok(), cgemmStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasCgemm success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op cgemm exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Cgemm Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeCgemmPlan(asdBlasHandle handle, asdBlasOperation_t transa, asdBlasOperation_t transb, int64_t m,
                                int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeCgemmPlan failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(m > 0 && n > 0 && k > 0, "blas asdBlasMakeCgemmPlan get m <= 0 || n <= 0 || k <= 0.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasCgemmPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasCgemmPlan({transa, transb, m, n, k, lda, ldb, ldc});
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Cgemm Plan failed: " << e.what();
        throw std::runtime_error("Make Cgemm Plan failed.");
    }
    if (plan->CreateTensor() != ErrorType::ACL_SUCCESS) {
        plan->MarkFailed();
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Fail to create blas asdBlasMakeCgemmPlan mask tensor.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
