/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/fp16_t.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "include/blasplan/BlasGemmBatchedPlan.h"


constexpr int ONE = 1;
constexpr int TWO = 2;

constexpr int64_t UB_SIZE = 192 * 1024;
constexpr uint32_t GATHER_OFFSETS_SIZE = 1024;

using namespace Mki;

namespace AsdSip {

// float16_t
BlasHCgemmBatchedPlan::BlasHCgemmBatchedPlan() : BlasPlan() {};

AsdSip::AspbStatus BlasHCgemmBatchedPlan::CreateTensor()
{
    int64_t shape[ONE] = {GATHER_OFFSETS_SIZE};
    int64_t stride[ONE] = {ONE};

    std::vector<uint32_t> offsets(GATHER_OFFSETS_SIZE, 0);
    for (size_t i = 0; i < offsets.size(); i++) {
        if (i % 2 == 0) {
            offsets[i] = sizeof(op::fp16_t) * (i + 1);
        } else {
            offsets[i] = sizeof(op::fp16_t) * (i - 1);
        }
    }
    void* deviceAddr{nullptr};
    auto ret = aclrtMalloc(&deviceAddr, GATHER_OFFSETS_SIZE * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ASDSIP_ECHECK(ret == 0, "Malloc gather-offsets in device failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    gatherOffsetsDeviceAddr = deviceAddr;

    // sync version
    ret = aclrtMemcpy(deviceAddr, GATHER_OFFSETS_SIZE * sizeof(uint32_t), offsets.data(), GATHER_OFFSETS_SIZE * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASDSIP_ECHECK(ret == 0, "Memcpy gather-offsets to device failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    gatherOffsets = aclCreateTensor(shape, ONE, aclDataType::ACL_UINT32,
        stride, 0, aclFormat::ACL_FORMAT_ND, shape, ONE, deviceAddr);
    ASDSIP_ECHECK(gatherOffsets != nullptr, "Create gather-offsets aclTensor failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasHCgemmBatchedPlan::FreeTensor()
{
    aclDestroyTensor(gatherOffsets);
    gatherOffsets = nullptr;
    if (gatherOffsetsDeviceAddr != nullptr) {
        aclrtFree(gatherOffsetsDeviceAddr);
        gatherOffsetsDeviceAddr = nullptr;
    }

    return ErrorType::ACL_SUCCESS;
}

BlasHCgemmBatchedPlan::~BlasHCgemmBatchedPlan()
{
    BlasPlan::DestroyPlanData();
}

int64_t BlasHCgemmBatchedPlan::GetWorkspaceSize()
{
    int64_t maxCubeCores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (maxCubeCores == 0) {
        maxCubeCores = 1;
    }

    int64_t remainUb = UB_SIZE - static_cast<int64_t>(GATHER_OFFSETS_SIZE * sizeof(uint32_t));
    int64_t useCubeCores = maxCubeCores;

    int64_t workspaceSize = remainUb * useCubeCores;
    ASDSIP_LOG(DEBUG) << "BlasHCgemmBatchedPlan estimates workspace-size:" << workspaceSize << ".\n";
    return workspaceSize;
}


// float
BlasCgemmBatchedPlan::BlasCgemmBatchedPlan() : BlasPlan() {};

AsdSip::AspbStatus BlasCgemmBatchedPlan::CreateTensor()
{
    int64_t shape[ONE] = {GATHER_OFFSETS_SIZE};
    int64_t stride[ONE] = {ONE};

    std::vector<uint32_t> offsets(GATHER_OFFSETS_SIZE, 0);
    for (size_t i = 0; i < offsets.size(); i++) {
        if (i % TWO == 0) {
            offsets[i] = sizeof(float) * (i + 1);
        } else {
            offsets[i] = sizeof(float) * (i - 1);
        }
    }
    void* deviceAddr{nullptr};
    auto ret = aclrtMalloc(&deviceAddr, GATHER_OFFSETS_SIZE * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ASDSIP_ECHECK(ret == 0, "Malloc gather-offsets in device failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    gatherOffsetsDeviceAddr = deviceAddr;

    // sync version
    ret = aclrtMemcpy(
        deviceAddr, GATHER_OFFSETS_SIZE * sizeof(uint32_t), offsets.data(), GATHER_OFFSETS_SIZE * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASDSIP_ECHECK(ret == 0, "Memcpy gather-offsets to device failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    gatherOffsets = aclCreateTensor(shape, ONE, aclDataType::ACL_UINT32,
        stride, 0, aclFormat::ACL_FORMAT_ND, shape, ONE, deviceAddr);
    ASDSIP_ECHECK(gatherOffsets != nullptr, "Create gather-offsets aclTensor failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasCgemmBatchedPlan::FreeTensor()
{
    aclDestroyTensor(gatherOffsets);
    gatherOffsets = nullptr;
    if (gatherOffsetsDeviceAddr != nullptr) {
        aclrtFree(gatherOffsetsDeviceAddr);
        gatherOffsetsDeviceAddr = nullptr;
    }

    return ErrorType::ACL_SUCCESS;
}

BlasCgemmBatchedPlan::~BlasCgemmBatchedPlan()
{
    BlasPlan::DestroyPlanData();
}

int64_t BlasCgemmBatchedPlan::GetWorkspaceSize()
{
    int64_t maxCubeCores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (maxCubeCores == 0) {
        maxCubeCores = 1;
    }

    int64_t remainUb = UB_SIZE - static_cast<int64_t>(GATHER_OFFSETS_SIZE * sizeof(uint32_t));
    int64_t useCubeCores = maxCubeCores;

    int64_t workspaceSize = remainUb * useCubeCores;
    ASDSIP_LOG(DEBUG) << "BlasCgemmBatchedPlan estimates workspace-size:" << workspaceSize << ".\n";
    return workspaceSize;
}

}
