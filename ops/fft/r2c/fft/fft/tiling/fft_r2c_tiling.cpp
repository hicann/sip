/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft_r2c_tiling.h"
#include "../../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "../../../../common/include/tiling/fft_all_mix_tiling_ops_utils.h"
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_r2c_arch35.h"
#include <mki/utils/platform/platform_info.h>

namespace AsdSip {
using namespace Mki;

// Allowed radices for decomposition (matches kernel's FindRadix)
static constexpr int64_t ALLOWED_RADICES[] = {2, 3, 5, 7};
static constexpr size_t NUM_RADICES = 4;

AspbStatus FftR2CTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    // Arch35 (Ascend 950): 使用专用 tiling 逻辑
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto& param = AnyCast<OpParam::FftR2CArch35>(launchParam.GetParam());

        FftAllMixTilingData* tilingDataPtr = reinterpret_cast<FftAllMixTilingData*>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                     return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        // 填充基础参数
        tilingDataPtr->batchSize = param.batchSize;
        tilingDataPtr->fftN = param.fftN;
        tilingDataPtr->radixListLen = param.radixListLen;
        tilingDataPtr->isInverse = param.isInverse;            // R2C: isInverse=0
        tilingDataPtr->isOddN = (param.fftN % 2 != 0) ? 1 : 0; // 新增：传递奇偶标志

        // 判断是否为奇数 N
        bool isOddN = (param.fftN % 2 != 0);

        // 重新计算 radix 分解 (与 fft_r2c_arch35.cpp::InitRadix 一致)
        // 偶数 N: 使用 N/2 分解（Pack 优化）
        // 奇数 N: 使用完整 N 分解（无 Pack）
        int64_t tempN = isOddN ? param.fftN : (param.fftN / 2);
        // 修复：按 Batch 划分的多核策略，每个 Block 独立处理 localBatch
        // currentBatch 应该从 1 开始，而不是从 param.batchSize 开始
        // Kernel 中 tempBatch 从 localBatchSize_ 开始累积
        int64_t currentBatch = 1;
        int64_t dftOffset = 0;
        int64_t twOffset = 0;

        for (int32_t s = 0; s < param.radixListLen; s++) {
            int64_t radix = 0;
            for (size_t i = 0; i < NUM_RADICES; i++) {
                if (tempN % ALLOWED_RADICES[i] == 0) {
                    radix = ALLOWED_RADICES[i];
                    break;
                }
            }

            // 防护: tempN 不含 2/3/5/7 因子（如素数 11/13/17）时 radix 为 0，
            // 下方 tempN/radix 会除零，直接返回参数错误（issue #117）
            if (radix == 0) {
                ASDSIP_LOG(ERROR) << "FftR2CTiling: fftN " << param.fftN
                                  << " contains prime factors other than 2/3/5/7, unsupported.";
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
            }

            int64_t M = tempN / radix;

            // 填充 plan 数组
            tilingDataPtr->radix_arr[s] = static_cast<int32_t>(radix);
            tilingDataPtr->M_arr[s] = M;
            tilingDataPtr->dft_offset_arr[s] = dftOffset;
            tilingDataPtr->tw_offset_arr[s] = twOffset;
            tilingDataPtr->currentBatch_arr[s] = currentBatch;

            // 更新偏移 (与 BuildFftPlan 中的计算一致)
            // W_R: 2*radix*radix floats, T: 2*radix*M floats
            dftOffset += 2 * radix * radix;
            twOffset += 2 * radix * M;

            currentBatch *= radix;
            tempN = M;
        }

        // 计算 workspace offsets (与 fft_r2c_arch35.cpp::EstimateWorkspaceSize 一致)
        // 偶数 N: N/2 点复数 FFT
        // 奇数 N: N 点复数 FFT
        int64_t fftPointCount = isOddN ? param.fftN : (param.fftN / 2);
        int64_t perBatchComplexBytes = fftPointCount * sizeof(float) * 2;
        int64_t workspaceSize = 2 * perBatchComplexBytes * param.batchSize; // ws0 + ws1

        tilingDataPtr->workspaceOffsets[0] = 0;                                      // ws0 offset
        tilingDataPtr->workspaceOffsets[1] = perBatchComplexBytes * param.batchSize; // ws1 offset
        tilingDataPtr->workspaceOffsets[2] = 0;
        tilingDataPtr->workspaceOffsets[3] = 0;
        tilingDataPtr->workspaceOffsets[4] = 0;

        // 设置多核数量 (使用 Vector Core 数量)
        uint32_t maxCore = static_cast<uint32_t>(
            Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_VECTOR));
        if (maxCore == 0) {
            maxCore = 1;
        }
        kernelInfo.SetBlockDim(maxCore);
        kernelInfo.GetScratchSizes().push_back(workspaceSize);

        ASDSIP_LOG(INFO) << "ASCEND_950 FftR2CTiling: batchSize=" << param.batchSize << ", fftN=" << param.fftN
                         << ", radixListLen=" << param.radixListLen << ", BlockDim=" << maxCore;

        return AsdSip::ErrorType::ACL_SUCCESS;
    }

    // 910B: 使用原有的 FftAllMix tiling 逻辑
    return InitFftAllMixTiling(launchParam, kernelInfo);
}
} // namespace AsdSip
