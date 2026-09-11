/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "strmm.h"
#include "blas_common.h"
#include "blasplan/include/blasplan/BlasStrmmPlan.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct StrmmTensorParam {
    const aclTensor* A;
    const aclTensor* B;
    const aclTensor* C;
};

AspbStatus StrmmDtypeCheck(struct StrmmTensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.B, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.C, aclDataType::ACL_FLOAT, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus StrmmShapeCheck(struct StrmmTensorParam parm, const int64_t m, const int64_t n, asdBlasSideMode_t side)
{
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasStrmm get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(m > 0 && m <= UINT32_MAX, "blas asdBlasStrmm get m <= 0 or m > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    int64_t aNum = (side == asdBlasSideMode_t::ASDBLAS_SIDE_LEFT) ? (m * m) : (n * n);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.A, aNum, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.B, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_NUM_NOT_MATCH(parm.C, m * n, ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasStrmm(asdBlasHandle handle, asdBlasSideMode_t side, asdBlasFillMode_t uplo, asdBlasOperation_t trans,
                        asdBlasDiagType_t diag, const int64_t m, const int64_t n, const float& alpha, aclTensor* A,
                        const int64_t lda, aclTensor* B, const int64_t ldb, aclTensor* C, const int64_t ldc)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Strmm get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct StrmmTensorParam parm {
        A, B, C
    };
    ASDSIP_CHECK(StrmmDtypeCheck(parm) == ErrorType::ACL_SUCCESS, "blas asdBlasStrmm dtype check failed.",
                 return ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    auto ret = StrmmShapeCheck(parm, m, n, side);
    ASDSIP_CHECK(ret == ErrorType::ACL_SUCCESS, "blas asdBlasStrmm shape check failed.", return ret);

    ASDSIP_ECHECK(diag == asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT, "blas asdBlasStrmm get diag unit.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    if (lda <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasStrmm get lda <= 0.";
    }

    if (ldb <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasStrmm get ldb <= 0.";
    }

    if (ldc <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasStrmm get ldc <= 0.";
    }

    try {
        AsdSip::BlasStrmmPlan& plan = dynamic_cast<AsdSip::BlasStrmmPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Strmm plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status strmmStatus;

        OpDesc opDesc;
        opDesc.opName = "StrmmOperation";
        AsdSip::OpParam::Strmm param;

        uint32_t realM = 0;
        uint32_t realN = 0;
        uint32_t realK = 0;
        uint32_t transLeft = 0;
        uint32_t transRight = 0;

        uint32_t sideInt = (side == asdBlasSideMode_t::ASDBLAS_SIDE_LEFT) ? 0 : 1;
        uint32_t uploInt = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER) ? 1 : 0;
        uint32_t lessFlag = 0;

        uint32_t caseOneFlag = 1;
        uint32_t caseTwoFlag = 2;
        uint32_t caseThreeFlag = 3;

        if (trans == asdBlasOperation_t::ASDBLAS_OP_C) {
            ASDSIP_LOG(DEBUG) << "trans before = " << static_cast<uint32_t>(trans);
            trans = asdBlasOperation_t::ASDBLAS_OP_T;
        }

        if (sideInt == 0 && uploInt == 0) {
            realM = static_cast<uint32_t>(m);
            realN = static_cast<uint32_t>(n);
            realK = static_cast<uint32_t>(m);
            transLeft = (trans == asdBlasOperation_t::ASDBLAS_OP_N) ? 0 : 1;
            transRight = 0;
        } else if (sideInt == 1 && uploInt == 0) {
            realM = static_cast<uint32_t>(m);
            realN = static_cast<uint32_t>(n);
            realK = static_cast<uint32_t>(n);
            transLeft = 0;
            transRight = (trans == asdBlasOperation_t::ASDBLAS_OP_N) ? 0 : 1;
            lessFlag = caseOneFlag;
        } else if (sideInt == 0 && uploInt == 1) {
            realM = static_cast<uint32_t>(m);
            realN = static_cast<uint32_t>(n);
            realK = static_cast<uint32_t>(m);
            transLeft = (trans == asdBlasOperation_t::ASDBLAS_OP_N) ? 0 : 1;
            transRight = 0;
            lessFlag = caseTwoFlag;
        } else if (sideInt == 1 && uploInt == 1) {
            realM = static_cast<uint32_t>(m);
            realN = static_cast<uint32_t>(n);
            realK = static_cast<uint32_t>(n);
            transLeft = 0;
            transRight = (trans == asdBlasOperation_t::ASDBLAS_OP_N) ? 0 : 1;
            lessFlag = caseThreeFlag;
        }

        param = {sideInt, uploInt, transLeft, transRight, 0, realM, realN, realK, lessFlag, alpha};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> strmmInTensors{};

        if (sideInt == 0) {
            strmmInTensors.push_back(A);
            strmmInTensors.push_back(B);
        } else {
            strmmInTensors.push_back(B);
            strmmInTensors.push_back(A);
        }

        SVector<aclTensor*> strmmOutTensors{C};

        strmmStatus = RunAsdOpsV2(plan.GetStream(), opDesc, strmmInTensors, strmmOutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(strmmStatus.Ok(), strmmStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasStrmm success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the Strmm exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Strmm Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeStrmmPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeStrmmPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // 重复绑定守卫：handle 只能初始化一次，防止 MakePlan 对已绑定 handle 静默失败导致 UAF（issue #129）
    ASDSIP_ECHECK(!BlasPlanCache::doesPlanExist(handle),
                  "blas handle already bound to a plan, repeated initialization is not allowed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    AsdSip::BlasStrmmPlan* plan = nullptr;
    try {
        plan = new AsdSip::BlasStrmmPlan();
        BlasPlanCache::MakePlan(handle, plan);
    } catch (const std::exception& e) {
        if (plan != nullptr) {
            delete plan;
        }
        delete static_cast<int*>(handle);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Strmm Plan failed: " << e.what();
        throw std::runtime_error("Make Strmm Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
