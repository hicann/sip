/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasCgemvBatchedPlan.h"
#include <iostream>
#include "mki/launch_param.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/assert.h"
#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

static constexpr uint8_t BUFFER_NUM = 2;
static constexpr uint32_t TMPBUF_NUM = 3;
static constexpr uint32_t ELENUM_LINE_ALIGNED = 32;  // max elements in line
static constexpr uint32_t UBUF_SIZE = 190 * 1024;    // 190kb
static constexpr uint32_t BYTENUM_BLOCK = 32;
static constexpr uint32_t BYTENUM_REPEAT = 256;
static constexpr uint32_t COMPLEX_ELENUM = 2;

static constexpr uint32_t FP16_DTYPE_SIZE = sizeof(float) / 2;  // 2b

static uint32_t CalMaxMatNum(bool isTrans, bool dataType, uint32_t m)
{
    uint32_t matNum = 0;
    if (isTrans) {
        if (dataType) {
            // trans && fp32
            matNum = (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM) * BYTENUM_REPEAT) /
                     (BUFFER_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED +
                         2 * BUFFER_NUM * sizeof(float) * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                         TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * BYTENUM_BLOCK);
        } else {
            // trans && fp16
            matNum =
                (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 2) * BYTENUM_REPEAT) /
                (BUFFER_NUM * FP16_DTYPE_SIZE * m * ELENUM_LINE_ALIGNED +
                    2 * BUFFER_NUM * FP16_DTYPE_SIZE * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                    TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * BYTENUM_BLOCK +
                    m * ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE + ELENUM_LINE_ALIGNED * sizeof(float));
        }
    } else {
        if (dataType) {
            // no trans && fp32
            matNum = (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 1) * BYTENUM_REPEAT) /
                     (BUFFER_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED +
                         2 * BUFFER_NUM * sizeof(float) * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                         TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * sizeof(float));
        } else {
            // no trans && fp16
            matNum =
                (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 3) * BYTENUM_REPEAT) /
                (BUFFER_NUM * FP16_DTYPE_SIZE * m * ELENUM_LINE_ALIGNED +
                    2 * BUFFER_NUM * FP16_DTYPE_SIZE * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                    TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * sizeof(float) +
                    m * ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE + ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE);
        }
    }
    matNum = matNum > 0 ? matNum : 1;
    return matNum;
}

namespace AsdSip {
using namespace Mki;
BlasCgemvBatchedPlan::BlasCgemvBatchedPlan(asdBlasOperation_t trans, asdDataType_t dtype, const int64_t m)
    : BlasPlan(), trans{trans}, dtype{dtype}, m{m}, maxMatNum{1}
{
    maskTensor.data = nullptr;
    maskTensor.hostData = nullptr;
}

AsdSip::AspbStatus BlasCgemvBatchedPlan::CreateTensor()
{
    SetMaskTensor();
    ASDSIP_ECHECK(maskAclTensor != nullptr, "BlasCgemvBatchedPlan create mask tensor failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);
    return ErrorType::ACL_SUCCESS;
}

void BlasCgemvBatchedPlan::SetMaskTensor()
{
    if (m <= 0) {
        m = 1;
    }

    bool isTrans = trans == asdBlasOperation_t::ASDBLAS_OP_N ? false : true;
    bool dataType = dtype == asdDataType_t::ASD_C_64F ? true : false;
    uint32_t dtypeSize = dataType ? sizeof(float) : FP16_DTYPE_SIZE;

    maxMatNum = CalMaxMatNum(isTrans, dataType, static_cast<uint32_t>(m));
    maxMatNum = maxMatNum > 0 ? maxMatNum : 1;

    uint32_t eleNumPerRepeat = BYTENUM_REPEAT / dtypeSize;
    uint32_t maskSize = maxMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM;
    uint32_t *maskData = nullptr;
    try {
        maskData = new uint32_t[maskSize];
    } catch (std::bad_alloc &e) {
        ASDSIP_LOG(ERROR) << "BlasCgemvBatchedPlan failed: " << e.what();
        return;
    }

    uint32_t realOffset = 0;
    uint32_t imagOffset = maxMatNum * ELENUM_LINE_ALIGNED + eleNumPerRepeat;

    int32_t k = 0;
    for (uint32_t i = 0; i < (maskSize / COMPLEX_ELENUM); i++) {
        maskData[k++] = (realOffset + i) * dtypeSize;
        maskData[k++] = (imagOffset + i) * dtypeSize;
    }

    maskTensor.desc = {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {maskSize}, {}, 0};
    maskTensor.hostData = maskData;
    maskTensor.dataSize = maskSize * sizeof(uint32_t);

    if (!MallocTensorInDevice(maskTensor).Ok()) {
        ASDSIP_LOG(ERROR) << "BlasCgemvBatchedPlan malloc mask tensor in device failed.";
        return;
    }
    toAclTensor(maskTensor, maskAclTensor);
    maskData = nullptr;
}

AsdSip::AspbStatus BlasCgemvBatchedPlan::FreeTensor()
{
    if (maskTensor.data != nullptr) {
        FreeTensorInDevice(maskTensor);
        maskTensor.data = nullptr;
    }
    if (maskTensor.hostData != nullptr) {
        delete[] static_cast<uint32_t *>(maskTensor.hostData);
        maskTensor.hostData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

BlasCgemvBatchedPlan::~BlasCgemvBatchedPlan()
{
    BlasPlan::DestroyPlanData();
}
}  // namespace AsdSip