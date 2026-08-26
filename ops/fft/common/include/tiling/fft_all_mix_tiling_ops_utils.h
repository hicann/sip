/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_FFT_ALL_MIX_TILING_OPS_UTILS
#define ASDOPS_FFT_ALL_MIX_TILING_OPS_UTILS

#include "fft_all_mix.h"
#include "fft_c2r_arch35.h"
#include "fft_all_mix_tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "utils/aspb_status.h"

namespace AsdSip {
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_THREE = 3;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t MAX_CORE_910B = 20;

using namespace Mki;
inline AsdSip::AspbStatus InitFftAllMixTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftC2RArch35>(launchParam.GetParam());
        FftAllMixTilingData* tilingDataPtr = reinterpret_cast<AsdSip::FftAllMixTilingData*>(
            kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->batchSize = param.batchSize;
        tilingDataPtr->fftN = param.fftN;
        tilingDataPtr->radixListLen = param.radixListLen;
        tilingDataPtr->isInverse = param.isInverse;

        // Precompute plan arrays on HOST side
        int64_t tempN = param.fftN;
        int64_t tempBatch = param.batchSize;
        int64_t dftOff = 0;
        int64_t twOff = 0;

        for (int32_t s = 0; s < param.radixListLen; s++) {
            // Compute radix
            int32_t radix = 0;
            if (tempN % 2 == 0)
                radix = 2;
            else if (tempN % 3 == 0)
                radix = 3;
            else if (tempN % 5 == 0)
                radix = 5;
            else if (tempN % 7 == 0)
                radix = 7;

            // 防护: tempN 不含 2/3/5/7 因子（如素数 11/13/17）时 radix 为 0，
            // 下方 tempN/radix 会除零，直接返回参数错误（issue #116）
            if (radix == 0) {
                ASDSIP_LOG(ERROR) << "InitFftAllMixTiling: fftN " << param.fftN
                                  << " contains prime factors other than 2/3/5/7, unsupported.";
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
            }

            tilingDataPtr->radix_arr[s] = radix;
            tilingDataPtr->M_arr[s] = tempN / radix;
            tilingDataPtr->currentBatch_arr[s] = tempBatch;
            tilingDataPtr->dft_offset_arr[s] = dftOff;
            tilingDataPtr->tw_offset_arr[s] = twOff;

            tempBatch *= radix;
            dftOff += 2 * radix * radix;
            twOff += 2 * radix * tilingDataPtr->M_arr[s];
            tempN = tilingDataPtr->M_arr[s];
        }

        int64_t singleBufferSize = param.batchSize * param.fftN * 2 * sizeof(float);
        int64_t workspaceSize = 2 * singleBufferSize; // ws0 + ws1

        tilingDataPtr->workspaceOffsets[0] = 0;
        tilingDataPtr->workspaceOffsets[1] = singleBufferSize;
        tilingDataPtr->workspaceOffsets[INDEX_TWO] = 0; // C2R不需要额外的workspace
        tilingDataPtr->workspaceOffsets[INDEX_THREE] = 0;
        tilingDataPtr->workspaceOffsets[INDEX_FOUR] = 0;

        uint32_t maxCore = 0;
        maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        if (maxCore == 0) {
            maxCore = 1;
        }

        kernelInfo.SetBlockDim(maxCore);
        kernelInfo.GetScratchSizes().push_back(workspaceSize);
        ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        const auto& param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());
        FftAllMixTilingData* tilingDataPtr = reinterpret_cast<AsdSip::FftAllMixTilingData*>(
            kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->batchSize = param.batchSize;
        tilingDataPtr->fftN = param.fftN;
        tilingDataPtr->radixListLen = param.radixListLen;
        tilingDataPtr->isInverse = param.isInverse;

        tilingDataPtr->workspaceOffsets[0] = 0;
        tilingDataPtr->workspaceOffsets[1] = tilingDataPtr->workspaceOffsets[0] + param.workspace_input_size;
        tilingDataPtr->workspaceOffsets[INDEX_TWO] = tilingDataPtr->workspaceOffsets[1] + param.workspace_output_size;
        tilingDataPtr->workspaceOffsets[INDEX_THREE] = tilingDataPtr->workspaceOffsets[INDEX_TWO] +
                                                       param.workspace_sync_size;
        tilingDataPtr->workspaceOffsets[INDEX_FOUR] = tilingDataPtr->workspaceOffsets[INDEX_THREE] +
                                                      param.workspace_c2c_output_size;

        uint32_t maxCore = 0;
        maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
        if (maxCore == 0) {
            maxCore = 1;
        }
        if (maxCore > MAX_CORE_910B) {
            maxCore = MAX_CORE_910B;
        }

        kernelInfo.SetBlockDim(maxCore);
        kernelInfo.GetScratchSizes().push_back(param.workspace_input_size + param.workspace_output_size +
                                               param.workspace_sync_size + param.workspace_c2c_output_size +
                                               param.workspace_auxil_size);
        ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

} // namespace AsdSip

#endif
