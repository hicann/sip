/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rot.h"
#include "blas_common.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {

constexpr int64_t COMPLEX_NUM = 2;

struct RotImplParam {
    int64_t elementCount;
    aclTensor* x;
    aclTensor* y;
    float c;
    float s;
};

AspbStatus asdBlasRotImpl(OpParam::Rot::RotType rotType, asdBlasHandle handle, RotImplParam implParam)
{
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Rot get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    int64_t* storageDims = nullptr;
    uint64_t rotStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.x, &storageDims, &rotStorageDimsNum),
                                 "asdBlasRotImpl: aclGetStorageShape");
    if (*storageDims * COMPLEX_NUM != implParam.elementCount) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasRotImpl get wrong x tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(implParam.y, &storageDims, &rotStorageDimsNum),
                                 "asdBlasRotImpl: aclGetStorageShape");
    if (*storageDims * COMPLEX_NUM != implParam.elementCount) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "blas asdBlasRotImpl get wrong y tensor num.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    try {
        AsdSip::BlasPlan& plan = dynamic_cast<AsdSip::BlasPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Rot plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        OpDesc opDesc;
        opDesc.opName = "RotOperation";
        AsdSip::OpParam::Rot param;
        param.elementCount = implParam.elementCount;
        param.cosValue = implParam.c;
        param.sinValue = implParam.s;
        param.rotType = rotType;
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> inTensors{implParam.x, implParam.y};
        SVector<aclTensor*> outTensors{};

        Status status = RunAsdOpsV2(plan.GetStream(), opDesc, inTensors, outTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasRot success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op rot exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Rot Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasCsrot(asdBlasHandle handle, const int64_t n, aclTensor* x, const int64_t incx, aclTensor* y,
                        const int64_t incy, const float& c, const float& s)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(x, &dataType), "asdBlasCsrot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCsrot get wrong x tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(y, &dataType), "asdBlasCsrot: aclGetDataType");
    ASDSIP_ECHECK(dataType == aclDataType::ACL_COMPLEX64, "blas asdBlasCsrot get wrong y tensor dtype.",
                  ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCsrot get incx <= 0.";
    }

    if (incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasCsrot get incy <= 0.";
    }
    ASDSIP_ECHECK(n > 0 && n <= UINT32_MAX, "blas asdBlasCsrot get n <= 0 or n > 2^32.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    float cosValue = c;
    float sinValue = s;
    OpParam::Rot::RotType rolType = OpParam::Rot::RotType::ROT_CSROT;
    RotImplParam param = {COMPLEX_NUM * n, x, y, cosValue, sinValue};
    return asdBlasRotImpl(rolType, handle, param);
}

AspbStatus asdBlasMakeRotPlan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeRotPlan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
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
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Rot Plan failed: " << e.what();
        throw std::runtime_error("Make Rot Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
