/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCaxpyPlan.h"
#include <complex>
#include "log/log.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "utils/aspb_status.h"

constexpr uint32_t K_FACTOR_4 = 4;
constexpr uint32_t K_FACTOR_38 = 38;
constexpr uint32_t K_FACTOR_1024 = 1024;
constexpr uint32_t COMPLEX_NUM = 2;

using namespace Mki;

namespace AsdSip {
BlasCaxpyPlan::BlasCaxpyPlan() : BlasPlan()
{
    augTensor.data = nullptr;
    augTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasCaxpyPlan::CreateTensor()
{
    uint32_t maxDataCount = K_FACTOR_38 * K_FACTOR_1024 / sizeof(float);
    uint32_t complexCount = maxDataCount / COMPLEX_NUM;

    uint32_t* augData = nullptr;
    try {
        augData = new uint32_t[maxDataCount];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasCaxpyPlan failed: " << e.what();
        ASDSIP_LOG(ERROR) << "malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    uint32_t offsetNum = COMPLEX_NUM;
    for (uint32_t i = 0; i < complexCount; i++) {
        augData[offsetNum * i] = K_FACTOR_4 * i;
        augData[offsetNum * i + 1] = K_FACTOR_4 * (i + complexCount);
    }

    augTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {maxDataCount}, {}, 0};
    augTensor.hostData = augData;
    augTensor.dataSize = maxDataCount * sizeof(uint32_t);

    if (!MallocTensorInDevice(augTensor).Ok()) {
        delete[] augData;
        augData = nullptr;
        augTensor.hostData = nullptr;
        ASDSIP_LOG(ERROR) << "augTensor malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    augData = nullptr;

    toAclTensor(augTensor, augAclTensor);
    if (augAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasCaxpyPlan::FreeTensor()
{
    if (augTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(augTensor.hostData);
        augTensor.hostData = nullptr;
    }
    if (augTensor.data != nullptr) {
        FreeTensorInDevice(this->augTensor);
        augTensor.data = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCaxpyPlan::~BlasCaxpyPlan() { BlasPlan::DestroyPlanData(); }

} // namespace AsdSip
