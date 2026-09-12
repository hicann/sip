/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCtrmvPlan.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

static constexpr uint32_t BYTE_NUM_FLOAT32 = 4;
constexpr uint32_t ELEMENTS_PER_COMPLEX = 2;
constexpr int64_t N0 = 64;

using namespace Mki;

namespace AsdSip {
BlasCtrmvPlan::BlasCtrmvPlan(asdBlasFillMode_t uplo, int64_t n) : BlasPlan(), uplo{uplo}, n{n}
{
    uploTensor.data = nullptr;
    uploTensor.hostData = nullptr;
};

AsdSip::AspbStatus BlasCtrmvPlan::CreateTensor() { return SetUploTensor(); }

AsdSip::AspbStatus BlasCtrmvPlan::SetUploTensor()
{
    int64_t blockSize = N0 * N0 * static_cast<int64_t>(sizeof(float));
    float ele = (uplo == asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER) ? 0 : 1;

    float* uploMatrixData = nullptr;
    try {
        uploMatrixData = new float[blockSize];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasCtrmvPlan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    for (int64_t i = 0; i < N0; ++i) {
        for (int64_t j = 0; j < N0; ++j) {
            if (j < i) {
                *(uploMatrixData + i * N0 + j) = ele;
            } else if (j == i) {
                *(uploMatrixData + i * N0 + j) = 1;
            } else {
                *(uploMatrixData + i * N0 + j) = 1 - ele;
            }
        }
    }

    uploTensor.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {N0, N0}, {}, 0};
    uploTensor.hostData = uploMatrixData;
    uploTensor.dataSize = static_cast<uint64_t>(blockSize);

    if (!MallocTensorInDevice(uploTensor).Ok()) {
        delete[] uploMatrixData;
        uploTensor.hostData = nullptr;
        uploMatrixData = nullptr;
        ASDSIP_LOG(ERROR) << "BlasCtrmvPlan uploTensor in device malloc failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(uploTensor, uploAclTensor);
    if (uploAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    uploMatrixData = nullptr;
    return ErrorType::ACL_SUCCESS;
};

AsdSip::AspbStatus BlasCtrmvPlan::FreeTensor()
{
    if (uploTensor.data != nullptr) {
        FreeTensorInDevice(uploTensor);
        uploTensor.data = nullptr;
    }
    if (uploTensor.hostData != nullptr) {
        delete[] static_cast<float*>(uploTensor.hostData);
        uploTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCtrmvPlan::~BlasCtrmvPlan() { BlasPlan::DestroyPlanData(); }

int64_t BlasCtrmvPlan::GetWorkspaceSize() { return n * ELEMENTS_PER_COMPLEX * sizeof(float); }
} // namespace AsdSip
