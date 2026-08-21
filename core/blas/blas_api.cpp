/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "blas_common.h"

namespace AsdSip {
AspbStatus asdBlasCreate(asdBlasHandle& handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    handle = BlasPlanCache::InitHandle();
    ASDSIP_LOG(INFO) << "BlasHandle create success.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasSetStream(asdBlasHandle handle, void* stream)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas plan does not exist.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ASDSIP_ECHECK(stream != nullptr, "stream is empty.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    BlasPlan& plan = BlasPlanCache::getPlan(handle);

    plan.SetStream(stream);
    ASDSIP_LOG(INFO) << "Blas set stream success.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasDestroy(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    if (!BlasPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "blas plan does not exist.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    BlasPlanCache::destroy_plan(handle);
    delete static_cast<int*>(handle);
    ASDSIP_LOG(INFO) << "BlasHandle destroy.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasSynchronize(asdBlasHandle handle)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas plan does not exist.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    BlasPlan& plan = BlasPlanCache::getPlan(handle);
    Mki::MkiRtStreamSynchronize(plan.GetStream());
    ASDSIP_LOG(INFO) << "Blas stream synchronized.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t& workspaceSize)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas plan does not exist.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    BlasPlan& plan = BlasPlanCache::getPlan(handle);
    workspaceSize = static_cast<size_t>(plan.GetWorkspaceSize());
    ASDSIP_LOG(INFO) << "Blas get workspaceSize success.";
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void* workSpace)
{
    std::lock_guard<std::mutex> lock(blas_mtx);
    ASDSIP_ECHECK(BlasPlanCache::doesPlanExist(handle), "blas plan does not exist.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    BlasPlan& plan = BlasPlanCache::getPlan(handle);
    plan.SetWorkspace((uint8_t*)workSpace);
    ASDSIP_LOG(INFO) << "Blas set workspace success.";
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
