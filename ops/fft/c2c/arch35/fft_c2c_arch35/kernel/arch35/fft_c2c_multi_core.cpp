/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t THREAD_NUM = 1024;

struct FftC2CArch35StageTilingData {
    int64_t batch;
    int64_t n;
    int64_t totalButterflies;
    int32_t nHalf;
    int32_t len;
    int32_t half;
    int32_t logNhalf;
    int32_t logHalf;
    int32_t twOffset;
    int32_t outer;
    int32_t isInverse;
    int32_t scaleOut;
    int32_t transpose;
};

template <bool SCALE_OUT, bool TRANSPOSE>
__simt_vf__ __launch_bounds__(THREAD_NUM) inline void StockhamRadix2SingleStage(
    __gm__ float* src, __gm__ float* twMat, __gm__ float* dst, int32_t n, int32_t nHalf, int32_t len, int32_t half,
    int32_t logNhalf, int32_t logHalf, int32_t twOffset, int32_t outer, int64_t totalButterflies, int64_t start,
    int64_t stride)
{
    for (int64_t i = start + threadIdx.x; i < totalButterflies; i += stride) {
        int64_t b = i >> logNhalf;
        int32_t idx = static_cast<int32_t>(i & (static_cast<int64_t>(nHalf) - 1));

        int32_t block = idx >> logHalf;
        int32_t j = idx & (half - 1);

        int64_t inBase = b * n + idx;

        int64_t uOff = inBase << 1;
        int64_t vOff = (inBase + nHalf) << 1;

        int32_t twOff = (twOffset + j) << 1;

        float u0 = src[uOff];
        float u1 = src[uOff + 1];

        float v0 = src[vOff];
        float v1 = src[vOff + 1];

        float w0 = twMat[twOff];
        float w1 = twMat[twOff + 1];

        float t0 = v0 * w0 - v1 * w1;
        float t1 = v0 * w1 + v1 * w0;

        float a0 = u0 + t0;
        float a1 = u1 + t1;

        float b0 = u0 - t0;
        float b1 = u1 - t1;

        if constexpr (SCALE_OUT) {
            float scale = 1.0f / n;
            a0 *= scale;
            a1 *= scale;
            b0 *= scale;
            b1 *= scale;
        }

        int64_t y0 = static_cast<int64_t>(block) * len + j;
        int64_t y1 = y0 + half;

        int64_t outIdx0;
        int64_t outIdx1;
        if constexpr (TRANSPOSE) {
            int64_t realBatch = b / outer;
            int64_t x = b - realBatch * outer;
            outIdx0 = (realBatch * n + y0) * outer + x;
            outIdx1 = (realBatch * n + y1) * outer + x;
        } else {
            int64_t batchBase = b * n;
            outIdx0 = batchBase + y0;
            outIdx1 = batchBase + y1;
        }

        int64_t outOff0 = outIdx0 << 1;
        int64_t outOff1 = outIdx1 << 1;

        dst[outOff0] = a0;
        dst[outOff0 + 1] = a1;

        dst[outOff1] = b0;
        dst[outOff1 + 1] = b1;
    }
}

template <bool SCALE_OUT>
__aicore__ inline void DispatchTranspose(__gm__ float* input, __gm__ float* twMat, __gm__ float* output,
                                         const FftC2CArch35StageTilingData& tilingData, int64_t start, int64_t stride)
{
    if (tilingData.transpose) {
        asc_vf_call<StockhamRadix2SingleStage<SCALE_OUT, true>>(
            dim3(THREAD_NUM), input, twMat, output, tilingData.n, tilingData.nHalf, tilingData.len, tilingData.half,
            tilingData.logNhalf, tilingData.logHalf, tilingData.twOffset, tilingData.outer, tilingData.totalButterflies,
            start, stride);
    } else {
        asc_vf_call<StockhamRadix2SingleStage<SCALE_OUT, false>>(
            dim3(THREAD_NUM), input, twMat, output, tilingData.n, tilingData.nHalf, tilingData.len, tilingData.half,
            tilingData.logNhalf, tilingData.logHalf, tilingData.twOffset, tilingData.outer, tilingData.totalButterflies,
            start, stride);
    }
}

extern "C" __global__ __vector__ void fft_c2c_arch35_multi_core(__gm__ float* input, __gm__ float* gmDftMatrixArray,
                                                                __gm__ float* twMat, __gm__ float* radixList,
                                                                __gm__ float* output, __gm__ float* workspace,
                                                                __gm__ uint8_t* gmTilingPara)
{
    (void)gmDftMatrixArray;
    (void)radixList;
    (void)workspace;

    auto tilingPtr = reinterpret_cast<__gm__ FftC2CArch35StageTilingData*>(gmTilingPara);
    FftC2CArch35StageTilingData tilingData;
    tilingData.batch = tilingPtr->batch;
    tilingData.n = tilingPtr->n;
    tilingData.totalButterflies = tilingPtr->totalButterflies;
    tilingData.nHalf = tilingPtr->nHalf;
    tilingData.len = tilingPtr->len;
    tilingData.half = tilingPtr->half;
    tilingData.logNhalf = tilingPtr->logNhalf;
    tilingData.logHalf = tilingPtr->logHalf;
    tilingData.twOffset = tilingPtr->twOffset;
    tilingData.outer = tilingPtr->outer;
    tilingData.isInverse = tilingPtr->isInverse;
    tilingData.scaleOut = tilingPtr->scaleOut;
    tilingData.transpose = tilingPtr->transpose;

    int64_t start = static_cast<int64_t>(THREAD_NUM) * AscendC::GetBlockIdx();
    int64_t stride = static_cast<int64_t>(AscendC::GetBlockNum()) * THREAD_NUM;

    if (tilingData.scaleOut) {
        DispatchTranspose<true>(input, twMat, output, tilingData, start, stride);
    } else {
        DispatchTranspose<false>(input, twMat, output, tilingData, start, stride);
    }
}
