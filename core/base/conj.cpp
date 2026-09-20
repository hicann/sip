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
#include "conj.h"

using namespace Mki;

namespace AsdSip {
AspbStatus Conj(const Tensor &inTensor, Tensor &outTensor, void *stream, uint8_t *workspace)
{
    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<Tensor> inTensors = {inTensor};
    SVector<Tensor> outTensors = {outTensor};

    Status status = RunAsdOps(stream, opDesc, inTensors, outTensors, workspace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    outTensor = outTensors.at(0);

    ASDSIP_LOG(INFO) << "Execute Conj success.";
    return ErrorType::ACL_SUCCESS;
}
}
