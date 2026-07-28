
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "convolve_tiling.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "convolve.h"
#include "convolve_tiling_data.h"
#include "convolve_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr uint64_t BASE_COL_BLOCK = 128;
constexpr uint64_t NUM_TWO = 2;

AsdSip::AspbStatus ConvolveTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    void* tilingData = kernelInfo.GetTilingHostAddr();

    ConvolveTilingData* tilingDataPtr = reinterpret_cast<ConvolveTilingData*>(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }

    OpParam::Convolve param = AnyCast<OpParam::Convolve>(launchParam.GetParam());

    uint64_t workspaceColNum = BASE_COL_BLOCK + static_cast<uint64_t>(param.kernelLen) * NUM_TWO - NUM_TWO;
    uint64_t workspaceRowNum = workspaceColNum + static_cast<uint64_t>(param.kernelLen) * NUM_TWO - NUM_TWO;

    tilingDataPtr->signalLen = param.signalLen;
    tilingDataPtr->kernelLen = param.kernelLen;
    tilingDataPtr->batchCount = param.batchCount;
    tilingDataPtr->dataType = param.dataType;

    // dataType: 0 = complex64 (sizeof(std::complex<float>) = 8), 1 = complex32 (sizeof(std::complex<half>) = 4)
    uint64_t dtypeSize = (param.dataType == 0) ? sizeof(float) * NUM_TWO : sizeof(float);
    uint64_t workspaceSize = workspaceColNum * workspaceRowNum * dtypeSize;
    kernelInfo.SetBlockDim(cubeCoreNum);
    kernelInfo.GetScratchSizes() = {workspaceSize};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
