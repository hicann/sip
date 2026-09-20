/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCgetriBatchedPlan.h"
#include <securec.h>
#include "mki/launch_param.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

static constexpr int64_t EYE_MATRIX_NUM = 3;
static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t TILE_ELENUM = 512;
static constexpr int64_t TILE_LENGTH = 4096;
static constexpr int64_t COMPLEX_ELENUM = 2;

static constexpr int64_t DTYPE_COMPLEX32 = 0;

namespace AsdSip {
BlasCgetriBatchedPlan::BlasCgetriBatchedPlan(int64_t n, int64_t batchSize, int64_t dtype)
    : BlasPlan(), n{n}, batchSize{batchSize}, dtype{dtype}
{
    wBatchTensor.data = nullptr;
    wBatchTensor.hostData = nullptr;
    gather1Offset.data = nullptr;
    gather1Offset.hostData = nullptr;
    gather2Offset.data = nullptr;
    gather2Offset.hostData = nullptr;
    gather3Offset.data = nullptr;
    gather3Offset.hostData = nullptr;
    eyeBatchMatTensor.data = nullptr;
    eyeBatchMatTensor.hostData = nullptr;
};


void BlasCgetriBatchedPlan::GenOffsetAndEyeMatData(std::vector<uint32_t> &gatherBuf1,
    std::vector<uint32_t> &gatherBuf2, std::vector<uint32_t> &gatherBuf3, std::vector<float> &eyeBuf) const
{
    int64_t tileM = TILE_ELENUM;
    int64_t N = this->n;
    int64_t blockM = tileM;
    int64_t blockN = BASE_BLOCK_ELENUM;
    int64_t strideN = (N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM;

    int64_t idx1 = 0;
    for (int64_t j = 0; j < blockN; ++j) {
        for (int64_t i = blockM - 1; i >= 0; --i) {
            gatherBuf1[idx1++] = static_cast<uint32_t>((i * blockN + j) * sizeof(float));
        }
    }

    int64_t idx2 = 0;
    for (int64_t i = blockM - 1; i >= 0; --i) {
        for (int64_t j = 0; j < blockN; ++j) {
            gatherBuf2[idx2++] = static_cast<uint32_t>((j * blockM + i) * sizeof(float));
        }
    }

    int64_t idx3 = 0;
    int64_t nn = std::min(N, TILE_LENGTH);
    int64_t mm = TILE_LENGTH / nn;
    for (int64_t i = 0; i < mm; ++i) {
        for (int64_t j = 0; j < nn; ++j) {
            gatherBuf3[idx3++] = static_cast<uint32_t>((i * nn + j) * sizeof(float));
            gatherBuf3[idx3++] = static_cast<uint32_t>((i * nn + j + TILE_LENGTH) * sizeof(float));
        }
    }

    int64_t batchNum = this->batchSize;
    int64_t eyeMatEleNum = static_cast<int64_t>(eyeBuf.size()) / batchNum;

    for (int64_t b = 0; b < batchNum; ++b) {
        for (int64_t i = 0; i < N; ++i) {
            eyeBuf[b * eyeMatEleNum + i * (strideN + 1)] = 1.0f;
        }
    }
}


void BlasCgetriBatchedPlan::GenBatchWAclTensor()
{
    int64_t blockN = BASE_BLOCK_ELENUM;
    int64_t N = this->n;
    int64_t M = this->n;
    int64_t batchNum = this->batchSize;

    // gen batch w data
    int64_t t = (std::min(M, N) + blockN - 1) / blockN * blockN;
    int64_t wEleNum = batchNum * t;
    std::vector<uint32_t> wBatchData(wEleNum, 0);
    memset_s(wBatchData.data(), wEleNum * sizeof(uint32_t), -1, wEleNum * sizeof(uint32_t));

    wBatchTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {wEleNum}, {}, 0};
    wBatchTensor.hostData = wBatchData.data();
    wBatchTensor.dataSize = wEleNum * sizeof(uint32_t);
    MallocTensorInDevice(wBatchTensor);
    toAclTensor(wBatchTensor, wBatchAcl);
}


void BlasCgetriBatchedPlan::GenWorkAclTensor()
{
    int64_t tileM = TILE_ELENUM;
    int64_t N = this->n;
    int64_t M = this->n;
    int64_t blockN = BASE_BLOCK_ELENUM;
    int64_t aMatWorkEleNum = ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM *
        COL_ALIGNED_ELENUM) * ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM *
        BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;

    int64_t workM = (M + tileM - 1) / tileM * tileM;
    int64_t workEleNum = workM * blockN * COMPLEX_ELENUM;

    std::vector<float> aWorkData(aMatWorkEleNum, 0.0f);
    std::vector<float> workGmData(workEleNum, 0.0f);

    aWorkTensor.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {aMatWorkEleNum}, {}, 0};
    aWorkTensor.hostData = aWorkData.data();
    aWorkTensor.dataSize = aMatWorkEleNum * sizeof(float);
    MallocTensorInDevice(aWorkTensor);
    toAclTensor(aWorkTensor, aWorkAcl);

    workGmTensor.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {workEleNum}, {}, 0};
    workGmTensor.hostData = workGmData.data();
    workGmTensor.dataSize = workEleNum * sizeof(float);
    MallocTensorInDevice(workGmTensor);
    toAclTensor(workGmTensor, workGmAcl);
}


AsdSip::AspbStatus BlasCgetriBatchedPlan::CreateTensor()
{
    GenBatchWAclTensor();
    GenWorkAclTensor();

    int64_t blockN = BASE_BLOCK_ELENUM;
    int64_t tileM = TILE_ELENUM;
    int64_t N = this->n;
    int64_t M = this->n;

    // generate offset and eye mat data
    int64_t gatherEleNum = tileM * blockN;
    int64_t eyeMatEleNum = ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM) *
        ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;
    int64_t batchNum = this->batchSize;

    std::vector<uint32_t> gatherOffset1(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset2(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset3(gatherEleNum, 0);
    std::vector<float> eyeBatchMatData(batchNum * eyeMatEleNum, 0.0f);

    GenOffsetAndEyeMatData(gatherOffset1, gatherOffset2, gatherOffset3, eyeBatchMatData);

    gather1Offset.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {gatherEleNum}, {}, 0};
    gather1Offset.hostData = gatherOffset1.data();
    gather1Offset.dataSize = gatherEleNum * sizeof(uint32_t);
    MallocTensorInDevice(gather1Offset);
    toAclTensor(gather1Offset, gather1OffsetAcl);

    gather2Offset.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {gatherEleNum}, {}, 0};
    gather2Offset.hostData = gatherOffset2.data();
    gather2Offset.dataSize = gatherEleNum * sizeof(uint32_t);
    MallocTensorInDevice(gather2Offset);
    toAclTensor(gather2Offset, gather2OffsetAcl);

    gather3Offset.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {gatherEleNum}, {}, 0};
    gather3Offset.hostData = gatherOffset3.data();
    gather3Offset.dataSize = gatherEleNum * sizeof(uint32_t);
    MallocTensorInDevice(gather3Offset);
    toAclTensor(gather3Offset, gather3OffsetAcl);

    eyeBatchMatTensor.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batchNum, eyeMatEleNum}, {}, 0};
    eyeBatchMatTensor.hostData = eyeBatchMatData.data();
    eyeBatchMatTensor.dataSize = batchNum * eyeMatEleNum * sizeof(float);
    MallocTensorInDevice(eyeBatchMatTensor);
    toAclTensor(eyeBatchMatTensor, eyeBatchMatAcl);

    return ErrorType::ACL_SUCCESS;
}


void BlasCgetriBatchedPlan::FreeTensorData(Tensor &input) const
{
    if (input.data != nullptr) {
        FreeTensorInDevice(input);
        input.data = nullptr;
    }
    if (input.hostData != nullptr) {
        input.hostData = nullptr;
    }
}


AsdSip::AspbStatus BlasCgetriBatchedPlan::FreeTensor()
{
    FreeTensorData(wBatchTensor);
    FreeTensorData(gather1Offset);
    FreeTensorData(gather2Offset);
    FreeTensorData(gather3Offset);
    FreeTensorData(eyeBatchMatTensor);

    FreeTensorData(aWorkTensor);
    FreeTensorData(workGmTensor);

    aclDestroyTensor(wBatchAcl);
    aclDestroyTensor(gather1OffsetAcl);
    aclDestroyTensor(gather2OffsetAcl);
    aclDestroyTensor(gather3OffsetAcl);
    aclDestroyTensor(eyeBatchMatAcl);

    aclDestroyTensor(aWorkAcl);
    aclDestroyTensor(workGmAcl);

    return ErrorType::ACL_SUCCESS;
}

BlasCgetriBatchedPlan::~BlasCgetriBatchedPlan()
{
    BlasPlan::DestroyPlanData();
}


int64_t BlasCgetriBatchedPlan::GetWorkspaceSize()
{
    int64_t workspaceSize = 0;
    int64_t rsvSize = 1024 * 1024; // for rsv
    if (this->dtype == DTYPE_COMPLEX32) {
        int64_t N = this->n;
        int64_t M = this->n;
        int64_t castWorkSize = N * M * sizeof(float) * COMPLEX_ELENUM;
        workspaceSize = castWorkSize + rsvSize;
    } else {
        workspaceSize = rsvSize;
    }
    ASDSIP_LOG(DEBUG) << "BlasCgetriBatchedPlan estimates workspace-size:" << workspaceSize << ".\n";
    return workspaceSize;
}
}