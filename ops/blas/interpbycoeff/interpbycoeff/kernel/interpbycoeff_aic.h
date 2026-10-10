/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPBYCOEFF_AIC_H
#define INTERPBYCOEFF_AIC_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "interpbycoeff_const.h"

using namespace AscendC;
using namespace matmul;

namespace InterpByCoeff {

template <typename T>
class InterpolationAIC {
public:
    __aicore__ inline InterpolationAIC(){};
    __aicore__ inline void Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                AsdSip::InterpByCoeffTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void MmCompute(int64_t offA, int64_t offN, int64_t matSizeN, int64_t offC);

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    MatmulImpl<aType, bType, cType, biasType> mm;

private:
    int64_t baseBlockIdx;

    int64_t batch;
    int64_t matSizeM;
    int64_t matSizeK;
    int64_t matSizeN;
    int64_t workspaceSize;
    int64_t tailBatch;
    int64_t blocksPerBatch;
    int64_t blockLength;
    int64_t usedCubeCoreNum;
    int64_t batchPerCore;
    int64_t splitN;
    int64_t splitLength;

    int64_t starBatch;
    int64_t starTailBatch;
    int64_t blocksPerCore = 0;
    int64_t tailblocks = 0;
    int64_t startN = 0;
    int64_t pingPongFlag = 0;
    int64_t totalCompute = 0;

private:
    __gm__ T* __restrict__ coeffGM;
    __gm__ T* __restrict__ workspaceGM;
    __gm__ T* __restrict__ outGM;

    GlobalTensor<T> matA; // 系数矩阵
    GlobalTensor<T> matB; // 插值关联点
    GlobalTensor<T> matC;
};

template <typename T>
__aicore__ inline void InterpolationAIC<T>::Init(InterpolationKernelParam kernelParam, GM_ADDR workspace,
                                                 AsdSip::InterpByCoeffTilingData* tilingData)
{
    baseBlockIdx = GetBlockIdx();

    batch = tilingData->batch;
    matSizeM = tilingData->interpLength;
    matSizeK = tilingData->rsNum * 2;
    matSizeN = tilingData->totalSubcarrier * 2;
    workspaceSize = tilingData->workspaceSize;
    tailBatch = tilingData->tailBatch;
    blocksPerBatch = tilingData->blocksPerBatch;
    blockLength = tilingData->blockLength * 2;
    usedCubeCoreNum = tilingData->usedCubeCoreNum;
    batchPerCore = tilingData->batchPerCore;
    splitN = tilingData->splitN;
    splitLength = tilingData->splitLength * 2;

    totalCompute = batchPerCore * splitN;
    starBatch = baseBlockIdx * batchPerCore;
    starTailBatch = batch - tailBatch;
    if (tailBatch != 0) {
        int64_t tailTotalBlockN = tailBatch * blocksPerBatch;
        blocksPerCore = tailTotalBlockN / usedCubeCoreNum;
        tailblocks = tailTotalBlockN % usedCubeCoreNum;
        if (baseBlockIdx < tailblocks) {
            blocksPerCore += 1;
            starTailBatch += blocksPerCore * baseBlockIdx / blocksPerBatch;
            startN = (blocksPerCore * baseBlockIdx % blocksPerBatch) * blockLength;
        } else {
            starTailBatch += (blocksPerCore * baseBlockIdx + tailblocks) / blocksPerBatch;
            startN = ((blocksPerCore * baseBlockIdx + tailblocks) % blocksPerBatch) * blockLength;
        }
        totalCompute += blocksPerCore;
    }

    coeffGM = (__gm__ T* __restrict__)(kernelParam.coeff);
    workspaceGM = (__gm__ T* __restrict__)workspace + workspaceSize * baseBlockIdx * 2 / sizeof(T);
    outGM = (__gm__ T* __restrict__)(kernelParam.out);
}

template <typename T>
__aicore__ inline void InterpolationAIC<T>::Process()
{
    int64_t coeffStridePerBatch = matSizeM * matSizeK;
    int64_t outStridePerBatch = matSizeM * matSizeN;
    if (batchPerCore != 0) {
        int64_t offsetA = starBatch * coeffStridePerBatch;
        int64_t offsetC = starBatch * outStridePerBatch;
        for (int i = 0; i < batchPerCore; i++) {
            int64_t nOffset = 0;
            for (int j = 0; j < splitN; j++) {
                int64_t length = nOffset + splitLength > matSizeN ? matSizeN - nOffset : splitLength;
                MmCompute(offsetA, nOffset, length, offsetC);
                nOffset += length;
            }
            offsetA += coeffStridePerBatch;
            offsetC += outStridePerBatch;
        }
    }

    if (tailBatch != 0) {
        int64_t offsetA = starTailBatch * coeffStridePerBatch;
        int64_t offsetC = starTailBatch * outStridePerBatch;
        int64_t nOffset = startN;
        for (int i = 0; i < blocksPerCore; i++) {
            if (nOffset + blockLength <= matSizeN) {
                MmCompute(offsetA, nOffset, blockLength, offsetC);
                nOffset += blockLength;
            } else {
                MmCompute(offsetA, nOffset, matSizeN - nOffset, offsetC);
                nOffset = 0;
                offsetA += coeffStridePerBatch;
                offsetC += outStridePerBatch;
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterpolationAIC<T>::MmCompute(int64_t offA, int64_t offN, int64_t singleN, int64_t offC)
{
    AscendC::CrossCoreWaitFlag(AIV_FINISH_FLAG_ID);

    matA.SetGlobalBuffer(coeffGM + offA);
    matB.SetGlobalBuffer(workspaceGM + workspaceSize * pingPongFlag / sizeof(T));
    matC.SetGlobalBuffer(outGM + offC + offN);

    mm.SetOrgShape(matSizeM, singleN, matSizeK, matSizeK, matSizeN);
    mm.SetSingleShape(matSizeM, singleN, matSizeK);
    mm.SetTensorA(matA);
    mm.SetTensorB(matB);
    mm.IterateAll(matC);
    mm.End();

    pingPongFlag = 1 - pingPongFlag;
    AscendC::CrossCoreSetFlag<SYNC_MODE_GROUP, PIPE_FIX>(AIC_FINISH_FLAG_ID);
}

} // namespace InterpByCoeff

#endif // INTERPBYCOEFF_AIC_H
