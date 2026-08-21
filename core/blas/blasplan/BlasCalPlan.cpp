/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCalPlan.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

constexpr uint32_t MAX_LENG_PER_UB_PROC = 6144; // 单次处理的最大数据量（复数） 6.75k ( 6912 -> 对齐32个complex元素)
constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr uint32_t PING_PONG_NUM = 2;
constexpr uint32_t BLAS_SCAL_WORKSPACE_SIZE = 16 * 1024;

using namespace Mki;

namespace AsdSip {
BlasCalPlan::BlasCalPlan() : m(0), n(0), incy(1), transA('N')
{
    maskTensor.data = nullptr;
    maskTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasCalPlan::CreateTensor() { return SetMaskTensor(); }

AsdSip::AspbStatus BlasCalPlan::SetMaskTensor()
{
    uint32_t imagBaseAddr = MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(float) +
                            MAX_LENG_PER_UB_PROC * sizeof(std::complex<float>) * PING_PONG_NUM;
    uint32_t realBaseAddr = imagBaseAddr + MAX_LENG_PER_UB_PROC * sizeof(uint32_t);

    uint32_t* maskData = nullptr;
    try {
        maskData = new uint32_t[MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasCalPlan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    int k = 0;
    for (uint32_t i = 0; i < MAX_LENG_PER_UB_PROC; i++) {
        maskData[k++] = realBaseAddr + i * sizeof(float);
        maskData[k++] = imagBaseAddr + i * sizeof(float);
    }
    maskTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {ELEMENTS_EACH_COMPLEX64, MAX_LENG_PER_UB_PROC}, {}, 0};
    maskTensor.hostData = maskData;
    maskTensor.dataSize = MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t);

    if (!MallocTensorInDevice(maskTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCalPlan maskTensor malloc failed";
        delete[] maskData;
        maskData = nullptr;
        maskTensor.hostData = nullptr;
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(maskTensor, maskAclTensor);
    return ErrorType::ACL_SUCCESS;
};

AsdSip::AspbStatus BlasCalPlan::FreeTensor()
{
    if (maskTensor.data != nullptr) {
        FreeTensorInDevice(maskTensor);
        maskTensor.data = nullptr;
    }
    if (maskTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(maskTensor.hostData);
        maskTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCalPlan::~BlasCalPlan() { BlasPlan::DestroyPlanData(); }

int64_t BlasCalPlan::GetWorkspaceSize() { return BLAS_SCAL_WORKSPACE_SIZE; }
} // namespace AsdSip
