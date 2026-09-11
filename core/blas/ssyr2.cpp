/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ssyr2.h"
#include <vector>
#include "blas_common.h"

using namespace AsdSip;
using namespace Mki;

namespace AsdSip {
struct Ssyr2TensorParam {
    const aclTensor* x;
    const aclTensor* y;
    const aclTensor* A;
};

AspbStatus Ssyr2DtypeCheck(struct Ssyr2TensorParam parm)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.A, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.x, aclDataType::ACL_FLOAT, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(parm.y, aclDataType::ACL_FLOAT, ret);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus Ssyr2ShapeCheck(struct Ssyr2TensorParam parm, const int64_t n)
{
    ASDSIP_ECHECK(n > 0, "blas asdBlasSsyr2 get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    SIP_OP_CHECK_SHAPE_NOT_MATCH(parm.A, (std::vector<int64_t>{n, n}), ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_SHAPE_NOT_MATCH(parm.x, (std::vector<int64_t>{n}), ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    SIP_OP_CHECK_SHAPE_NOT_MATCH(parm.y, (std::vector<int64_t>{n}), ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasSsyr2(asdBlasHandle handle, asdBlasFillMode_t uplo, const int64_t n, const float& alpha, aclTensor* x,
                        int64_t incx, aclTensor* y, int64_t incy, aclTensor* A, const int64_t lda)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas Ssyr2 get cached plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    struct Ssyr2TensorParam parm {
        x, y, A
    };

    auto ret = Ssyr2DtypeCheck(parm);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    ret = Ssyr2ShapeCheck(parm, n);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    if (incx <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSsyr2 get incx <= 0.";
        incx = 1;
    }

    if (incy <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSsyr2 get incy <= 0.";
        incy = 1;
    }

    if (lda <= 0) {
        ASDSIP_LOG(INFO) << "blas asdBlasSsyr2 get lda <= 0.";
    }

    try {
        AsdSip::BlasPlan& plan = dynamic_cast<AsdSip::BlasPlan&>(BlasPlanCache::getPlan(handle));
        ASDSIP_ECHECK(plan.IsInitialized(), "Ssyr2 plan init Error!.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

        Status status;

        // A = alpha * x * y.T + alpha * x.T * y + A
        OpDesc opDesc;
        opDesc.opName = "Ssyr2Operation";
        AsdSip::OpParam::Ssyr2 param;

        uint32_t uploLocal = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER) ? 0 : 1;
        param = {uploLocal, static_cast<uint32_t>(n), static_cast<uint32_t>(incx), static_cast<uint32_t>(incy), alpha};
        opDesc.specificParam = param;
        ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

        SVector<aclTensor*> ssyr2InTensors{x, y};
        SVector<aclTensor*> ssyr2OutTensors{A};

        status = RunAsdOpsV2(plan.GetStream(), opDesc, ssyr2InTensors, ssyr2OutTensors, plan.GetWorkspace());
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

        ASDSIP_LOG(INFO) << "Execute asdBlasSsyr2 success.";
        return ErrorType::ACL_SUCCESS;
    } catch (std::bad_cast& e) {
        // Handle the op ssyr2 exception
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Ssyr2 Error: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
}

AspbStatus asdBlasMakeSsyr2Plan(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(handle != nullptr, "blas MakeSsyr2Plan Fail.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
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
        ASDSIP_ELOG(ErrorType::ACL_ERROR_INTERNAL_ERROR) << "Make Ssyr2 Plan failed: " << e.what();
        throw std::runtime_error("Make Ssyr2 Plan failed.");
    }
    plan->MarkInitialized();
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
