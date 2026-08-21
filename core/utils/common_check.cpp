/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <optional>
#include "utils/common_check.h"
#include "utils/assert.h"

using namespace AsdSip;

std::optional<int64_t> GetTensorNum(uint64_t dimsNum, int64_t* storageDims)
{
    if (storageDims == nullptr) {
        return std::nullopt;
    }
    int64_t elementNum = 1;
    for (size_t i = 0; i < dimsNum; i++) {
        elementNum *= storageDims[i];
    }
    return elementNum;
}

AsdSip::AspbStatus AsdTensorDtypeCheck(const aclTensor* tensor, const aclDataType expectedDtype)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(tensor, &dataType), "aclGetDataType");
    if (dataType != expectedDtype) {
        ASDSIP_LOG(ERROR) << "Tensor expected dtype is " << expectedDtype << " , but found is: " << dataType;
        return AsdSip::ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus AsdTensorShapeCheck(const aclTensor* tensor, const std::vector<int64_t> expectedDims)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    uint64_t expectedDimNum = expectedDims.size();
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(tensor, &storageDims, &storageDimsNum), "aclGetStorageShape");
    if (storageDimsNum != expectedDimNum) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_LOG(ERROR) << "Tensor expected dim num is " << expectedDimNum << " , but found is: " << storageDimsNum;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < expectedDimNum; i++) {
        if (*(storageDims + i) != expectedDims[i]) {
            ASDSIP_LOG(ERROR) << "Tensor dim [" << i << "] value is  " << *(storageDims + i)
                              << " , which is not as expected";
            delete[] storageDims;
            storageDims = nullptr;
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    }

    delete[] storageDims;
    storageDims = nullptr;
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus AsdTensorNumCheck(const aclTensor* tensor, int64_t expectedNum)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(tensor, &storageDims, &storageDimsNum), "aclGetStorageShape");
    std::optional<int64_t> numOpt = GetTensorNum(storageDimsNum, storageDims);
    if (!numOpt.has_value() || numOpt.value() != expectedNum) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_LOG(ERROR) << "Tensor num is " << (numOpt.has_value() ? std::to_string(numOpt.value()) : "invalid")
                          << " , which is not as expected.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    delete[] storageDims;
    storageDims = nullptr;
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus AsdTensorInvalidShapeCheck(const aclTensor* tensor)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(tensor, &storageDims, &storageDimsNum), "aclGetStorageShape");
    // 防护: 0 维张量(rank=0)时 storageDims 可能为 nullptr, 直接解引用会崩溃
    if (storageDims == nullptr || storageDimsNum == 0) {
        ASDSIP_LOG(ERROR) << "Tensor shape is invalid, storage dims is empty.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        if (storageDims[i] <= 0) {
            delete[] storageDims;
            storageDims = nullptr;
            ASDSIP_LOG(ERROR) << "Tensor shape is invalid.";
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
        }
    }

    delete[] storageDims;
    storageDims = nullptr;
    return AsdSip::ErrorType::ACL_SUCCESS;
}
