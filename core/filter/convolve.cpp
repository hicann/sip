/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/assert.h"
#include "log/log.h"
#include "filter_api.h"
#include "utils/ops_base.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/common_types.h"
#include "convolve.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {

constexpr uint64_t DIMS_TWO = 2;
constexpr uint64_t BASE_COL_BLOCK = 128;

AsdSip::AspbStatus asdConvolve(const aclTensor* signal, const aclTensor* kernel, aclTensor* output,
                               asdConvolveMode_t mode, void* stream, void* workspace)
{
    (void)mode;
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(
        aclGetStorageShape(static_cast<const aclTensor*>(signal), &storageDims, &storageDimsNum),
        "asdConvolve: aclGetStorageShape");
    if (*storageDims <= 0 || storageDimsNum > DIMS_TWO) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdConvolve get wrong inputs.";
        return AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    int64_t batchCount = storageDims[0];
    int64_t signalLen = storageDims[1];

    delete[] storageDims;
    storageDims = nullptr;

    CHECK_STATUS_WITH_ACL_RETURN(
        aclGetStorageShape(static_cast<const aclTensor*>(kernel), &storageDims, &storageDimsNum),
        "asdConvolve: aclGetStorageShape");
    if (*storageDims <= 0 || storageDimsNum > DIMS_TWO) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "asdConvolve get wrong inputs.";
        return AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }
    int64_t kernelLen = storageDims[0];

    delete[] storageDims;
    storageDims = nullptr;

    op::DataType dataType = op::DataType::DT_UNDEFINED;
    dataType = signal->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
                  "asdConvolve get wrong signal tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = kernel->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_FLOAT || dataType == op::DataType::DT_FLOAT16,
                  "asdConvolve get wrong kernel tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    dataType = op::DataType::DT_UNDEFINED;
    dataType = output->GetDataType();
    ASDSIP_ECHECK(dataType == op::DataType::DT_COMPLEX64 || dataType == op::DataType::DT_COMPLEX32,
                  "asdConvolve get wrong output tensor dtype.", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    OpDesc opDesc;
    opDesc.opName = "ConvolveOperation";
    AsdSip::OpParam::Convolve param;
    param.batchCount = batchCount;
    param.signalLen = signalLen;
    param.kernelLen = kernelLen;
    param.dataType = (dataType == op::DataType::DT_COMPLEX64) ? 0 : 1;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor*> inTensors{const_cast<aclTensor*>(static_cast<const aclTensor*>(signal)),
                                  const_cast<aclTensor*>(static_cast<const aclTensor*>(kernel))};
    SVector<aclTensor*> outTensors{const_cast<aclTensor*>(static_cast<const aclTensor*>(output))};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, reinterpret_cast<uint8_t*>(workspace));
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute asdConvolve success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdConvolveGetWorkspaceSize(int64_t signalLen, int64_t kernelLen, size_t& size)
{
    ASDSIP_ECHECK(signalLen > 0 && kernelLen > 0, "asdConvolveGetWorkspaceSize get invalid params",
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    size_t workspaceColNum = BASE_COL_BLOCK + static_cast<size_t>(kernelLen * DIMS_TWO - DIMS_TWO);
    size_t workspaceRowNum = workspaceColNum + static_cast<size_t>(kernelLen * DIMS_TWO - DIMS_TWO);
    size = workspaceColNum * workspaceRowNum;
    size = size * sizeof(float);

    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
