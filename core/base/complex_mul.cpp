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
#include "base_api.h"
#include "utils/ops_base.h"
#include "mul.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/common_types.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
static AspbStatus GetStorageElementCount(const aclTensor *tensor, int64_t &elementCount)
{
    int64_t *storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetStorageShape(tensor, &storageDims, &storageDimsNum), "asdMul: aclGetStorageShape");
    int64_t count = 1;
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        if (storageDims[i] <= 0) {
            count = 0;
            break;
        }
        count *= storageDims[i];
    }
    delete[] storageDims;
    elementCount = count;
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdMul(int n, const aclTensor *x, const aclTensor *y, aclTensor *z, void *stream, void *workspace)
{
    int64_t sizeX = 0;
    CHECK_STATUS_WITH_ACL_RETURN(GetStorageElementCount(x, sizeX), "asdMul: aclGetStorageShape");
    int64_t sizeY = 0;
    CHECK_STATUS_WITH_ACL_RETURN(GetStorageElementCount(y, sizeY), "asdMul: aclGetStorageShape");
    int64_t sizeZ = 0;
    CHECK_STATUS_WITH_ACL_RETURN(GetStorageElementCount(z, sizeZ), "asdMul: aclGetStorageShape");

    op::DataType dataTypeX = op::DataType::DT_UNDEFINED;
    dataTypeX = x->GetDataType();
    ASDSIP_ECHECK(dataTypeX == op::DataType::DT_COMPLEX64 || dataTypeX == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong x tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    op::DataType dataTypeY = op::DataType::DT_UNDEFINED;
    dataTypeY = y->GetDataType();
    ASDSIP_ECHECK(dataTypeY == op::DataType::DT_COMPLEX64 || dataTypeY == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong y tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    op::DataType dataTypeZ = op::DataType::DT_UNDEFINED;
    dataTypeZ = z->GetDataType();
    ASDSIP_ECHECK(dataTypeZ == op::DataType::DT_COMPLEX64 || dataTypeZ == op::DataType::DT_COMPLEX32,
        "base asdMul get wrong z tensor dtype.",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(dataTypeX == dataTypeY && dataTypeY == dataTypeZ,
        "Input x, y, z tensors must have the same data type",
        ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);

    ASDSIP_ECHECK(n > 0, "base asdMul get n <= 0.", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(n == sizeX && n == sizeY && n == sizeZ,
        "Invalid tensor dimensions: parameter 'n' must be equal to sizeX, sizeY, and sizeZ.",
        ErrorType::ACL_ERROR_INVALID_PARAM);

    OpDesc opDesc;
    opDesc.opName = "MulOperation";
    AsdSip::OpParam::CMul param;
    param.n = n;
    param.cMulType = (dataTypeX == op::DataType::DT_COMPLEX64) ? OpParam::CMul::CMulType::MUL_C64
                     : OpParam::CMul::CMulType::MUL_C32;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<aclTensor *> inTensors{const_cast<aclTensor *>(x), const_cast<aclTensor *>(y)};
    SVector<aclTensor *> outTensors{z};

    Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, (uint8_t *)workspace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    ASDSIP_LOG(INFO) << "Execute asdBaseMul success.";
    return ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip