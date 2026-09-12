/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasIamaxPlan.h"
#include <securec.h>
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

using namespace Mki;

namespace AsdSip {
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

BlasIamaxPlan::BlasIamaxPlan() : BlasPlan()
{
    zeroTensor.data = nullptr;
    zeroTensor.hostData = nullptr;
};

BlasIamaxPlan::~BlasIamaxPlan() { BlasPlan::DestroyPlanData(); }

AsdSip::AspbStatus BlasIamaxPlan::CreateTensor()
{
    int32_t* zeroData = nullptr;
    try {
        zeroData = new int32_t[1];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasIamaxPlan failed: " << e.what();
        ASDSIP_LOG(ERROR) << "malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    zeroData[0] = static_cast<int32_t>(1);

    zeroTensor.desc = {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}, {}, 0};
    zeroTensor.hostData = zeroData;
    zeroTensor.dataSize = 1 * sizeof(int32_t);
    if (!MallocTensorInDevice(zeroTensor).Ok()) {
        delete[] zeroData;
        zeroTensor.hostData = nullptr;
        zeroData = nullptr;
        ASDSIP_LOG(ERROR) << "zeroTensor in device malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(zeroTensor, zeroAclTensor);
    if (zeroAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    zeroData = nullptr;
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasIamaxPlan::FreeTensor()
{
    if (zeroTensor.data != nullptr) {
        FreeTensorInDevice(zeroTensor);
        zeroTensor.data = nullptr;
    }
    if (zeroTensor.hostData != nullptr) {
        delete[] static_cast<int32_t*>(zeroTensor.hostData);
        zeroTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
