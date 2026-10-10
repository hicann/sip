/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPBYCOEFF_AIV_H
#define INTERPBYCOEFF_AIV_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "interpbycoeff_const.h"

using namespace AscendC;
using namespace matmul;

namespace InterpByCoeff {

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_BLOCK = 32;
constexpr int32_t BYTES_PER_REPEAT = 256;

template <typename T>
class InterpolationAIV {
public:
    __aicore__ inline InterpolationAIV(){};
    __aicore__ inline void Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                AsdSip::InterpByCoeffTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t batchOffset, int64_t nOffset, int64_t length);
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

private:
    int64_t blockIdx;
    int64_t subblockIdx;
    int64_t baseBlockIdx;

    int64_t batch;
    int64_t rs;
    int64_t totalSubcarrier;
    int64_t interpLength;
    int64_t batchPerCore;
    int64_t tailBatch;
    int64_t splitN;
    int64_t splitLength;
    int64_t blocksPerBatch;
    int64_t blockLength;
    int64_t usedCubeCoreNum;
    int64_t workspaceSize;

    int64_t tailTotalBlockN;
    int64_t batchStride;
    int64_t dataBlockSize;

    int64_t subRs;
    int64_t subOffset;
    int64_t pingPongFlag = 0;
    bool isFirst = true;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> calcQueue;
    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> workspcaeGM;
    GlobalTensor<T> outGM;
    LocalTensor<uint32_t> gatherOffset;
};

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                                 AsdSip::InterpByCoeffTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    subblockIdx = blockIdx % 2;
    baseBlockIdx = blockIdx / 2;

    batch = tilingData->batch;
    rs = tilingData->rsNum;
    totalSubcarrier = tilingData->totalSubcarrier * 2;
    interpLength = tilingData->interpLength;
    batchPerCore = tilingData->batchPerCore;
    tailBatch = tilingData->tailBatch;
    splitN = tilingData->splitN;
    splitLength = tilingData->splitLength * 2;
    blocksPerBatch = tilingData->blocksPerBatch;
    blockLength = tilingData->blockLength * 2;
    usedCubeCoreNum = tilingData->usedCubeCoreNum;
    workspaceSize = tilingData->workspaceSize;

    batchStride = rs * totalSubcarrier;
    tailTotalBlockN = tailBatch * blocksPerBatch;
    subRs = rs / 2;
    subOffset = 0;
    if (subblockIdx == 1) {
        subOffset += subRs;
        subRs = rs - subRs;
    }

    inTensorsGM.SetGlobalBuffer((__gm__ T*)kernelParam.x);
    workspcaeGM.SetGlobalBuffer((__gm__ T*)workspace + baseBlockIdx * workspaceSize * 2 / sizeof(T));
    outGM.SetGlobalBuffer((__gm__ T*)kernelParam.out);
    pipe.InitBuffer(dataQueue, BUFFER_NUM, splitLength * sizeof(T) * subRs);
    pipe.InitBuffer(outQueue, BUFFER_NUM, splitLength * sizeof(T) * subRs * 2);
    pipe.InitBuffer(calcQueue, splitLength * sizeof(uint32_t));
    gatherOffset = calcQueue.AllocTensor<uint32_t>();
    for (uint32_t idx = 0; idx < 8; idx++) {
        if (idx % 2 == 0) {
            gatherOffset.SetValue(idx, (idx + 1) * sizeof(T));
        } else {
            gatherOffset.SetValue(idx, (idx - 1) * sizeof(T));
        }
    }
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    auto tmpTensor = gatherOffset.template ReinterpretCast<int32_t>();
    if (splitLength > 8) {
        int32_t addNum = 8 * sizeof(T);
        for (uint32_t idx = 8; idx < splitLength;) {
            int32_t addNumCount = (2 * idx > splitLength) ? splitLength - idx : idx;
            Adds(tmpTensor[idx], tmpTensor, addNum, addNumCount);
            PipeBarrier<PIPE_V>();
            idx += addNumCount;
            addNum = 2 * addNum;
        }
    }
}

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Process()
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

    if (batchPerCore != 0) {
        int64_t batchOffset = baseBlockIdx * batchPerCore;
        for (int i = 0; i < batchPerCore; i++) {
            int64_t nOffset = 0;
            for (int j = 0; j < splitN; j++) {
                int64_t length = nOffset + splitLength > totalSubcarrier ? totalSubcarrier - nOffset : splitLength;
                Compute(batchOffset, nOffset, length);
                nOffset += length;
            }
            batchOffset++;
        }
    }

    if (tailBatch != 0) {
        int blocksPerCore = tailTotalBlockN / usedCubeCoreNum;
        int tailblocks = tailTotalBlockN % usedCubeCoreNum;
        int64_t batchOffset = batch - tailBatch;
        int64_t nOffset = 0;
        if (baseBlockIdx < tailblocks) {
            blocksPerCore += 1;
            batchOffset += blocksPerCore * baseBlockIdx / blocksPerBatch;
            nOffset = (blocksPerCore * baseBlockIdx % blocksPerBatch) * blockLength;
        } else {
            batchOffset += (blocksPerCore * baseBlockIdx + tailblocks) / blocksPerBatch;
            nOffset = ((blocksPerCore * baseBlockIdx + tailblocks) % blocksPerBatch) * blockLength;
        }
        for (int i = 0; i < blocksPerCore; i++) {
            if (nOffset + blockLength <= totalSubcarrier) {
                Compute(batchOffset, nOffset, blockLength);
                nOffset += blockLength;
            } else {
                Compute(batchOffset, nOffset, totalSubcarrier - nOffset);
                nOffset = 0;
                batchOffset += 1;
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterpolationAIV<T>::Compute(int64_t batchOffset, int64_t nOffset, int64_t length)
{
    int64_t offset = batchOffset * batchStride + nOffset + subOffset * totalSubcarrier;
    // copy in
    LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();
    uint16_t dataBlock = BYTE_BLOCK / sizeof(T);
    uint32_t calCount = (length + dataBlock - 1) / dataBlock * dataBlock;
    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = subRs;
    dataCopyExtParams.blockLen = length * sizeof(T);
    dataCopyExtParams.srcStride = (totalSubcarrier - length) * sizeof(T);
    dataCopyExtParams.dstStride = 0;
    DataCopyPadExtParams<T> padExtParams{false, 0, 0, 0};
    DataCopyPad(dataLocal, inTensorsGM[offset], dataCopyExtParams, padExtParams);
    dataQueue.EnQue(dataLocal);
    // Compute
    dataLocal = dataQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    uint64_t mask = BYTES_PER_REPEAT / sizeof(T);
    uint64_t mulMask[2] = {6148914691236517205, 6148914691236517205};
    if constexpr (IsSameType<T, float>::value) {
        mulMask[1] = 0;
    }
    uint64_t repeatTimes = (calCount * sizeof(T) + BYTES_PER_REPEAT - 1) / BYTES_PER_REPEAT;
    for (int i = 0; i < subRs; i++) {
        Copy(outLocal[i * 2 * calCount], dataLocal[i * calCount], mask, repeatTimes, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Gather(outLocal[(i * 2 + 1) * calCount], dataLocal[i * calCount], gatherOffset, 0, calCount);
        PipeBarrier<PIPE_V>();
        Muls(outLocal[(i * 2 + 1) * calCount], outLocal[(i * 2 + 1) * calCount], T(-1), mulMask, repeatTimes,
             {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    dataQueue.FreeTensor(dataLocal);
    outQueue.EnQue<T>(outLocal);
    // copy out
    outLocal = outQueue.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = subRs * 2;
    extParams.blockLen = length * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(workspcaeGM[pingPongFlag * workspaceSize / sizeof(T) + subOffset * length * 2], outLocal, extParams);
    outQueue.FreeTensor(outLocal);
    pingPongFlag = 1 - pingPongFlag;
    AscendC::CrossCoreSetFlag<SYNC_MODE_GROUP, PIPE_MTE3>(AIV_FINISH_FLAG_ID);
    if (isFirst) {
        isFirst = false;
        return;
    }
    AscendC::CrossCoreWaitFlag(AIC_FINISH_FLAG_ID);
}

} // namespace InterpByCoeff
#endif // INTERPBYCOEFF_AIV_H
