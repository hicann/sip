/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasColwiseMulPlan.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;
constexpr uint32_t MAX_DATA_COUNT = 32 * 1024 / sizeof(float);

using namespace Mki;

namespace AsdSip {
BlasColwiseMulPlan::BlasColwiseMulPlan() : n(0)
{
    augTensor.data = nullptr;
    augTensor.hostData = nullptr;
};

BlasColwiseMulPlan::~BlasColwiseMulPlan() { BlasPlan::DestroyPlanData(); }

AsdSip::AspbStatus BlasColwiseMulPlan::CreateTensor()
{
    uint32_t complexCount = MAX_DATA_COUNT / COMPLEX_NUM;

    uint32_t* augData = nullptr;
    try {
        augData = new uint32_t[MAX_DATA_COUNT];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasColwiseMulPlan failed: " << e.what();
        ASDSIP_LOG(ERROR) << "malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    for (uint32_t i = 0; i < complexCount; i++) {
        augData[COMPLEX_NUM * i] = FP32_BYTE_SIZE * i;
        augData[COMPLEX_NUM * i + 1] = FP32_BYTE_SIZE * (i + complexCount);
    }

    augTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {MAX_DATA_COUNT}, {}, 0};
    augTensor.hostData = augData;
    augTensor.dataSize = MAX_DATA_COUNT * sizeof(uint32_t);

    if (!MallocTensorInDevice(augTensor).Ok()) {
        delete[] augData;
        augData = nullptr;
        augTensor.hostData = nullptr;
        ASDSIP_LOG(ERROR) << "augTensor in device malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(augTensor, augAclTensor);
    if (augAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    augData = nullptr;
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasColwiseMulPlan::FreeTensor()
{
    if (augTensor.data != nullptr) {
        FreeTensorInDevice(this->augTensor);
        augTensor.data = nullptr;
    }
    if (augTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(augTensor.hostData);
        augTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
