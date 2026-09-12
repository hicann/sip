/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "dft.h"
#include "dft_tiling_data.h"
#include "dft_tiling.h"

namespace AsdSip {

using Mki::AnyCast;
using Mki::CoreType;
using Mki::KernelInfo;
using Mki::LaunchParam;
using Mki::PlatformInfo;
using Mki::PlatformType;
using Mki::Status;
using Mki::TENSOR_DTYPE_COMPLEX32;
using Mki::TENSOR_DTYPE_COMPLEX64;

constexpr uint32_t DFT_SIZE_MULTIPLIER = 2;
constexpr uint32_t WORKSPACE_SIZE = 32;
constexpr uint32_t MAX_WORKSPACE_SIZE = 192 * 1024 * 40 * 4; // UB * 40
constexpr int64_t UB_SIZE = 192 * 1024;
constexpr uint32_t SUB_BLOCK_NUM = 2;

AsdSip::AspbStatus DftTilingForComplex32(DftTilingData* tilingDataPtr, uint32_t maxCore)
{
    int32_t maxBatchPreCore = static_cast<int32_t>(UB_SIZE / 3 / sizeof(int16_t) / tilingDataPtr->n);
    int32_t vecNum = static_cast<int32_t>(maxCore * SUB_BLOCK_NUM);
    int32_t maxBatchPreLoop = maxBatchPreCore * vecNum;
    if (tilingDataPtr->m > maxBatchPreLoop) {
        tilingDataPtr->loopTime = (tilingDataPtr->m + maxBatchPreLoop - 1) / maxBatchPreLoop;
        tilingDataPtr->batchNumPreLoop = maxBatchPreLoop;
        tilingDataPtr->batchNumPreCore = maxBatchPreCore;
        tilingDataPtr->batchTailNum = tilingDataPtr->m - maxBatchPreLoop * (tilingDataPtr->loopTime - 1);
        tilingDataPtr->batchTailNumPreCore = tilingDataPtr->batchTailNum / vecNum;
        tilingDataPtr->batchTailCoreNum = tilingDataPtr->batchTailNum % vecNum;
    } else {
        tilingDataPtr->loopTime = 1;
        tilingDataPtr->batchNumPreLoop = 0;
        tilingDataPtr->batchNumPreCore = 0;
        tilingDataPtr->batchTailNum = tilingDataPtr->m;
        tilingDataPtr->batchTailNumPreCore = tilingDataPtr->batchTailNum / vecNum;
        tilingDataPtr->batchTailCoreNum = tilingDataPtr->batchTailNum % vecNum;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus DftTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    const auto& param = AnyCast<OpParam::Dft>(launchParam.GetParam());

    DftTilingData* tilingDataPtr = reinterpret_cast<AsdSip::DftTilingData*>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    const int thresholdK = 32768;
    const int cubeDataCountPerLoopSmall = 128;
    const int cubeDataCountPerLoopLarge = 64;

    tilingDataPtr->batchSize = 1;
    tilingDataPtr->m = param.batchSize;
    tilingDataPtr->n = param.fftN * DFT_SIZE_MULTIPLIER;
    tilingDataPtr->k = param.fftN * DFT_SIZE_MULTIPLIER;
    tilingDataPtr->transA = 0;
    tilingDataPtr->transB = 0;
    const auto platformType = PlatformInfo::Instance().GetPlatformType();
    if (platformType == PlatformType::ASCEND_910B || platformType == PlatformType::ASCEND_950) {
        if (static_cast<bool>(param.isInverse)) {
            tilingDataPtr->transB = 1;
        }
    }
    tilingDataPtr->m0 = cubeDataCountPerLoopSmall;
    tilingDataPtr->n0 = cubeDataCountPerLoopSmall;
    tilingDataPtr->k0 = cubeDataCountPerLoopSmall;
    if (tilingDataPtr->k > thresholdK) {
        tilingDataPtr->m0 = cubeDataCountPerLoopLarge;
        tilingDataPtr->n0 = cubeDataCountPerLoopLarge;
        tilingDataPtr->k0 = cubeDataCountPerLoopLarge;
    }
    uint32_t maxCore = 0;
    maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (maxCore == 0) {
        maxCore = 1;
    }

    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX32) {
        DftTilingForComplex32(tilingDataPtr, maxCore);
    }

    kernelInfo.SetBlockDim(maxCore);
    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64) {
        kernelInfo.GetScratchSizes().push_back(WORKSPACE_SIZE);
    } else {
        kernelInfo.GetScratchSizes().push_back(MAX_WORKSPACE_SIZE);
    }

    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
