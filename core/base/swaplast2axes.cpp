/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <optional>
#include "utils/assert.h"
#include "log/log.h"
#include "utils/common_check.h"
#include "base_api.h"
#include "utils/ops_base.h"
#include "swap_last_2axes.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

AspbStatus SwapLast2AxesDtypeCheck(const aclTensor* inTensor, aclTensor* outTensor)
{
    auto ret = ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE;
    SIP_OP_CHECK_DTYPE_NOT_MATCH(inTensor, aclDataType::ACL_COMPLEX64, ret);
    SIP_OP_CHECK_DTYPE_NOT_MATCH(outTensor, aclDataType::ACL_COMPLEX64, ret);

    return ErrorType::ACL_SUCCESS;
}

AspbStatus SwapLast2AxesShapeCheck(const aclTensor* inTensor, aclTensor* outTensor, int64_t*& storageDims,
                                   uint64_t& inStorageDimsNum)
{
    uint64_t outStorageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(outTensor, &storageDims, &outStorageDimsNum),
                                 "swapLast2Axes: aclGetStorageShape");
    std::optional<int64_t> outNumOpt = GetTensorNum(outStorageDimsNum, storageDims);

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(inTensor, &storageDims, &inStorageDimsNum),
                                 "swapLast2Axes: aclGetStorageShape");

    // element num check
    std::optional<int64_t> inNumOpt = GetTensorNum(inStorageDimsNum, storageDims);
    if (!outNumOpt.has_value() || !inNumOpt.has_value() || outNumOpt.value() < inNumOpt.value()) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_LOG(ERROR) << "swapLast2Axes get wrong inTensor or outTensor num."
                          << "In element num is " << inNumOpt.value_or(-1) << " , and out element num is "
                          << outNumOpt.value_or(-1) << " . Out num must be no less than in num";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // dim: 2 or 3 && inDim = outDim
    if (!(inStorageDimsNum == outStorageDimsNum && inStorageDimsNum > 1 && inStorageDimsNum < 4)) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_LOG(ERROR) << "blas swapLast2Axes get wrong inTensor tensor.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    return ErrorType::ACL_SUCCESS;
}

AspbStatus swapLast2Axes(const aclTensor* inTensor, aclTensor* outTensor, void* stream, void* workspace)
{
    auto ret = SwapLast2AxesDtypeCheck(inTensor, outTensor);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    int64_t* storageDims = nullptr;
    uint64_t inStorageDimsNum = 0;
    ret = SwapLast2AxesShapeCheck(inTensor, outTensor, storageDims, inStorageDimsNum);
    if (ret != ErrorType::ACL_SUCCESS) {
        return ret;
    }

    OpDesc opDesc;
    opDesc.opName = "SwapLast2AxesOperation";
    AsdSip::OpParam::SwapLast2Axes param;
    int64_t dim = static_cast<int64_t>(inStorageDimsNum);
    param.batch = 1;
    int64_t endDimIdx = 2;
    int64_t frontDimIdx = 1;
    for (int64_t i = 0; i < dim - endDimIdx; i++) {
        param.batch *= static_cast<uint32_t>(storageDims[i]);
    }
    param.row = static_cast<uint32_t>(storageDims[dim - endDimIdx]);
    param.col = static_cast<uint32_t>(storageDims[dim - frontDimIdx]);
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> inTensors = {const_cast<aclTensor*>(inTensor)};
    SVector<aclTensor*> outTensors = {outTensor};

    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, (uint8_t*)workspace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute swapLast2Axes success.";
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus swapLast2AxesGetWorkspaceSize(size_t& size)
{
    size = ASYNC_WORKSPACE_SIZE;
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
