/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "include/blasplan/BlasComplexMatDotPlan.h"
#include <complex>
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

using namespace Mki;

// 27kb
constexpr uint32_t MAX_DATA_COUNT = 27 * 1024 / sizeof(float);

constexpr uint32_t MUL_NUM = 2;

constexpr uint32_t FOUR_NUM = 4;

namespace AsdSip {
BlasComplexMatDotPlan::BlasComplexMatDotPlan() : BlasPlan(), n(0)
{
    augDotTensor.data = nullptr;
    augDotTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasComplexMatDotPlan::CreateTensor()
{
    uint32_t complexCount = MAX_DATA_COUNT / 2;

    uint32_t* augData = nullptr;
    try {
        augData = new uint32_t[MAX_DATA_COUNT];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasComplexMatDotPlan failed: " << e.what();
        ASDSIP_LOG(ERROR) << "malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    for (uint32_t i = 0; i < complexCount; i++) {
        augData[MUL_NUM * i] = FOUR_NUM * i;
        augData[MUL_NUM * i + 1] = FOUR_NUM * (i + complexCount);
    }

    augDotTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {MAX_DATA_COUNT}, {}, 0};
    augDotTensor.hostData = augData;
    augDotTensor.dataSize = MAX_DATA_COUNT * sizeof(uint32_t);

    if (!MallocTensorInDevice(augDotTensor).Ok()) {
        delete[] augData;
        augData = nullptr;
        augDotTensor.hostData = nullptr;
        ASDSIP_LOG(ERROR) << "augTensor in device malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(augDotTensor, augAclTensor);
    if (augAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    augData = nullptr;
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasComplexMatDotPlan::FreeTensor()
{
    if (augDotTensor.data != nullptr) {
        FreeTensorInDevice(this->augDotTensor);
        augDotTensor.data = nullptr;
    }
    if (augDotTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(augDotTensor.hostData);
        augDotTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasComplexMatDotPlan::~BlasComplexMatDotPlan() { BlasPlan::DestroyPlanData(); }
} // namespace AsdSip
