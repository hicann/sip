/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "BlasPlan.h"
namespace AsdSip {
using namespace Mki;

class BlasCgetriBatchedPlan : public BlasPlan {
public:
    explicit BlasCgetriBatchedPlan(int64_t n, int64_t batchSize, int64_t dtype);
    ~BlasCgetriBatchedPlan() override;
    int64_t GetWorkspaceSize() override;
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus FreeTensor() override;

    aclTensor *wBatchAcl = nullptr;
    aclTensor *gather1OffsetAcl = nullptr;
    aclTensor *gather2OffsetAcl = nullptr;
    aclTensor *gather3OffsetAcl = nullptr;
    aclTensor *eyeBatchMatAcl = nullptr;

    aclTensor *aWorkAcl = nullptr;
    aclTensor *workGmAcl = nullptr;

private:
    void GenBatchWAclTensor();
    void GenWorkAclTensor();
    void GenOffsetAndEyeMatData(std::vector<uint32_t> &gatherBuf1, std::vector<uint32_t> &gatherBuf2,
        std::vector<uint32_t> &gatherBuf3, std::vector<float> &eyeBuf) const;
    void FreeTensorData(Tensor &input) const;

    int64_t n;
    int64_t batchSize;
    int64_t dtype;
    Tensor wBatchTensor;
    Tensor gather1Offset;
    Tensor gather2Offset;
    Tensor gather3Offset;
    Tensor eyeBatchMatTensor;

    Tensor aWorkTensor;
    Tensor workGmTensor;
};
}
