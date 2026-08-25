/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCgemvPlan.h"
#include "mki/launch_param.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

static constexpr uint32_t BYTE_NUM_FLOAT32 = 4;
namespace AsdSip {
using namespace Mki;
constexpr size_t CALC_DIV_TWO = 2;
BlasCgemvPlan::BlasCgemvPlan(asdBlasOperation_t trans, const int64_t m, const int64_t n, aclTensor* y,
                             const int64_t incy)
    : BlasPlan(), trans{trans}, m{m}, n{n}, incy{incy}, inputY{y}
{
    maskTensor.data = nullptr;
    maskTensor.hostData = nullptr;
    yInTensor.hostData = nullptr;
    yInTensor.data = nullptr;
    SetyInTensor(y);
};

AsdSip::AspbStatus BlasCgemvPlan::CreateTensor()
{
    if (SetyInTensor(inputY) == ErrorType::ACL_SUCCESS) {
        return SetMaskTensor();
    }
    return ErrorType::ACL_ERROR_INTERNAL_ERROR;
}

AsdSip::AspbStatus BlasCgemvPlan::SetMaskTensor()
{
    if (incy <= 0) {
        incy = 1;
    }

    if (m <= 0) {
        m = 1;
    }

    if (n <= 0) {
        n = 1;
    }

    uint32_t realOffset = 0;
    uint32_t imagOffset = 1024;

    uint32_t maskSize = imagOffset * 2;
    uint32_t* maskData = nullptr;
    try {
        maskData = new uint32_t[maskSize];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "BlasCgemvPlan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    int k = 0;
    for (uint32_t i = 0; i < (maskSize / CALC_DIV_TWO); i++) {
        maskData[k++] = (realOffset + i) * BYTE_NUM_FLOAT32;
        maskData[k++] = (imagOffset + i) * BYTE_NUM_FLOAT32;
    }

    maskTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {maskSize}, {}, 0};
    maskTensor.hostData = maskData;
    maskTensor.dataSize = maskSize * sizeof(uint32_t);

    if (!MallocTensorInDevice(maskTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCgemvPlan maskTensor malloc failed: ";
        delete[] maskData;
        maskData = nullptr;
        maskTensor.hostData = nullptr;
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    toAclTensor(maskTensor, maskAclTensor);
    if (maskAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclnTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    maskData = nullptr;
    return ErrorType::ACL_SUCCESS;
};

AsdSip::AspbStatus BlasCgemvPlan::SetyInTensor(aclTensor* y)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    auto ret = aclGetStorageShape(y, &storageDims, &storageDimsNum);
    if (ret != 0 || storageDims == nullptr || *storageDims <= 0) {
        if (storageDims != nullptr) {
            delete[] storageDims;
            storageDims = nullptr;
        }
        ASDSIP_LOG(ERROR) << "blas asdBlasCgemv get wrong y tensor num.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    int64_t ySize = sizeof(std::complex<float>);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        ySize *= storageDims[i];
    }

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    yInTensor.desc = {
        TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {static_cast<int64_t>(ySize / sizeof(std::complex<float>))}, {}, 0};
    yInTensor.dataSize = static_cast<uint64_t>(ySize);

    // 构造函数与 CreateTensor 均会调用 SetyInTensor，重复分配前必须释放旧设备内存，
    // 否则首次分配的指针被覆盖造成泄漏（issue #115）
    if (yInTensor.data != nullptr) {
        FreeTensorInDevice(yInTensor);
        yInTensor.data = nullptr;
    }

    ret = aclrtMalloc(&yInTensor.data, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
        ASDSIP_LOG(ERROR) << "Device malloc y_in failed.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    ret = aclrtMemcpy(yInTensor.data, ySize, Mki::GetStorageAddr(y), ySize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != 0) {
        ASDSIP_LOG(ERROR) << "Memcpy input y device to device y_in failed.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    toAclTensor(yInTensor, yInAclTensor);
    if (yInAclTensor == nullptr) {
        ASDSIP_LOG(ERROR) << "convert to aclnTensor fail.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    return ErrorType::ACL_SUCCESS;
};

AsdSip::AspbStatus BlasCgemvPlan::FreeTensor()
{
    if (maskTensor.data != nullptr) {
        FreeTensorInDevice(maskTensor);
        maskTensor.data = nullptr;
    }
    if (maskTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t*>(maskTensor.hostData);
        maskTensor.hostData = nullptr;
    }
    if (yInTensor.data != nullptr) {
        FreeTensorInDevice(yInTensor);
        yInTensor.data = nullptr;
    }
    if (yInTensor.hostData != nullptr) {
        free(yInTensor.hostData);
        yInTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCgemvPlan::~BlasCgemvPlan() { BlasPlan::DestroyPlanData(); }
} // namespace AsdSip
