/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCgercPlan.h"
#include <complex>
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

using namespace Mki;

constexpr uint32_t MAX_DATACOUNT = 32 * 1024 / 4;
constexpr uint32_t NUM_FLAG = 2;
constexpr uint32_t NUM_INBYTES = 4;
namespace AsdSip {
BlasCgercPlan::BlasCgercPlan() : BlasPlan()
{
    gatherOffset.data = nullptr;
    gatherOffset.hostData = nullptr;
};

AsdSip::AspbStatus BlasCgercPlan::CreateTensor()
{
    uint32_t gatherOffsetSize = MAX_DATACOUNT;

    uint32_t* offsetData = nullptr;
    try {
        offsetData = new uint32_t[gatherOffsetSize];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasCgercPlan failed: " << e.what();
        ASDSIP_LOG(ERROR) << "malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    for (uint32_t i = 0; i < gatherOffsetSize / NUM_FLAG; i++) {
        offsetData[NUM_FLAG * i] = NUM_INBYTES * i;
        offsetData[NUM_FLAG * i + 1] = NUM_INBYTES * (i + gatherOffsetSize / NUM_FLAG);
    }

    gatherOffset.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {gatherOffsetSize}, {}, 0};
    gatherOffset.hostData = offsetData;
    gatherOffset.dataSize = gatherOffsetSize * sizeof(uint32_t);

    if (!MallocTensorInDevice(gatherOffset).Ok()) {
        delete[] offsetData;
        offsetData = nullptr;
        gatherOffset.hostData = nullptr;
        ASDSIP_LOG(ERROR) << "gatherOffset in device malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(gatherOffset, gatherAclOffset);
    if (gatherAclOffset == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    offsetData = nullptr;

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasCgercPlan::FreeTensor()
{
    if (gatherOffset.data != nullptr) {
        FreeTensorInDevice(this->gatherOffset);
        gatherOffset.data = nullptr;
    }
    if (gatherOffset.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(gatherOffset.hostData);
        gatherOffset.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCgercPlan::~BlasCgercPlan() { BlasPlan::DestroyPlanData(); }
} // namespace AsdSip
