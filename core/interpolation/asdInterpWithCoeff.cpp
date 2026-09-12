/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interp_api.h"
#include "utils/ops_base.h"
#include "utils/op_desc.h"
#include "utils/aspb_status.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki/tensor.h"
#include "mki/utils/rt/rt.h"
#include "mki/utils/assert/assert.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "interpbycoeff.h"

using namespace AsdSip;
using namespace Mki;

constexpr uint32_t INTERP_WORKSPACE_SIZE = 65536 * 20 * 2;
constexpr uint64_t INTERP_DIMS_THREE = 3;
constexpr int64_t MAX_BATCH = 1024;
constexpr int64_t INTERP_RS_TWO = 2;
constexpr int64_t INTERP_RS_FOUR = 4;
constexpr int64_t MAX_TOTAL_SUBCARRIER = 32760;
constexpr int64_t MAX_SIGNAL_NUM = 14;
constexpr int DIMS_TWO = 2;

namespace AsdSip {
int64_t* getInputShape(const aclTensor* x)
{
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    auto ret = aclGetStorageShape(x, &storageDims, &storageDimsNum);
    if (ret != ACL_SUCCESS || *storageDims <= 0 || storageDimsNum != INTERP_DIMS_THREE) {
        if (storageDims != nullptr) {
            delete[] storageDims;
            storageDims = nullptr;
        }
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "interpByCoeff get wrong input tensor.";
        return nullptr;
    }
    if (storageDims[0] > MAX_BATCH || (storageDims[1] != INTERP_RS_TWO && storageDims[1] != INTERP_RS_FOUR) ||
        storageDims[DIMS_TWO] > MAX_TOTAL_SUBCARRIER) {
        delete[] storageDims;
        storageDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "interpByCoeff do not support input tensor shape.";
        return nullptr;
    }
    return storageDims;
}

int64_t* getCoeffShape(const aclTensor* coefficient)
{
    int64_t* coeffDims = nullptr;
    uint64_t coeffDimsNum = 0;
    auto ret = aclGetStorageShape(coefficient, &coeffDims, &coeffDimsNum);
    if (ret != ACL_SUCCESS || *coeffDims <= 0 || coeffDimsNum != INTERP_DIMS_THREE || coeffDims[1] < 0 ||
        coeffDims[1] > MAX_SIGNAL_NUM) {
        delete[] coeffDims;
        coeffDims = nullptr;
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "interpByCoeff get wrong coefficient tensor.";
    }
    return coeffDims;
}

void cleanAcl(int64_t* storageDims, int64_t* coeffDims)
{
    if (storageDims != nullptr) {
        delete[] storageDims;
        storageDims = nullptr;
    }
    if (coeffDims != nullptr) {
        delete[] coeffDims;
        coeffDims = nullptr;
    }
}

AspbStatus asdInterpWithCoeff(const aclTensor* x, const aclTensor* coefficient, aclTensor* output, void* stream,
                              void* workSpace)
{
    int64_t* storageDims = getInputShape(x);
    ASDSIP_ECHECK(storageDims != nullptr, "InterpWithCoeff failed.", ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    int64_t* coeffDims = getCoeffShape(coefficient);
    ASDSIP_ECHECK(coeffDims != nullptr, "InterpWithCoeff failed.", ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);
    if (storageDims[0] != coeffDims[0] || storageDims[1] != coeffDims[DIMS_TWO]) {
        cleanAcl(storageDims, coeffDims);
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "input and coefficient do not match.";
        return ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    OpDesc opDesc;
    opDesc.opName = "InterpByCoeffOperation";
    AsdSip::OpParam::InterpByCoeff param;
    param.batch = storageDims[0];
    param.rsNum = storageDims[1];
    param.totalSubcarrier = storageDims[DIMS_TWO];
    param.interpLength = coeffDims[1];
    opDesc.specificParam = param;

    cleanAcl(storageDims, coeffDims);

    SVector<aclTensor*> inTensors{const_cast<aclTensor*>(x), const_cast<aclTensor*>(coefficient)};
    SVector<aclTensor*> outTensors{output};
    Mki::Status status = RunAsdOpsV2(stream, opDesc, inTensors, outTensors, (uint8_t*)workSpace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    output = outTensors.at(0);

    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdInterpWithCoeffGetWorkspaceSize(size_t& workspaceSize)
{
    workspaceSize = INTERP_WORKSPACE_SIZE;
    return ErrorType::ACL_SUCCESS;
}

} // namespace AsdSip
