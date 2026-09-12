/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fft_regbase_tiling.h"
#include "fft_regbase_tiling_data.h"
#include "params/fft_c2c_regbase.h"
#include <mki/utils/platform/platform_info.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "log/log.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;

AspbStatus FftC2CRegBaseTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    ASDSIP_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_950 &&
                     launchParam.GetParam().Type() == typeid(OpParam::FftC2CRegBase),
                 "RegBase requires Ascend 950 and valid parameters", return ErrorType::ACL_ERROR_INVALID_PARAM);
    const auto& param = AnyCast<OpParam::FftC2CRegBase>(launchParam.GetParam());
    ASDSIP_CHECK(OpParam::IsFftC2CRegBaseSupported(param.fftN) && param.batchSize > 0, "RegBase shape is unsupported",
                 return ErrorType::ACL_ERROR_INVALID_PARAM);
    auto* tiling = reinterpret_cast<FftC2CRegBaseTilingData*>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tiling != nullptr, "RegBase tiling allocation is missing", return ErrorType::ACL_ERROR_INVALID_PARAM);
    const uint32_t aivCount = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    ASDSIP_CHECK(aivCount > 0, "RegBase requires a positive physical AIV count",
                 return ErrorType::ACL_ERROR_INVALID_PARAM);

    const int32_t n = static_cast<int32_t>(param.fftN);
    const int32_t tilePoints = n <= 256 ? 256 : std::max(1024, n);
    const int32_t coreGrainPoints = n <= 256 ? 128 : 1024;
    const int64_t targetBatchPerCore = std::max<int64_t>(1, (coreGrainPoints + n - 1) / n);
    const int64_t neededCores = 1 + (param.batchSize - 1) / targetBatchPerCore;
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(aivCount, neededCores));
    const bool doubleBuffer = n <= 4096 && param.batchSize > static_cast<int64_t>(aivCount) * (tilePoints / n);
    *tiling = {param.batchSize, n, tilePoints};
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(static_cast<uint64_t>(2 * n + (doubleBuffer ? 1 : 0)));
    // MKI packs one pointer per scratch entry before tiling. Keep this
    // zero-byte placeholder so the declared workspace argument cannot shift
    // the tiling pointer to the overflow-argument position.
    kernelInfo.GetScratchSizes() = {0};

    const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
    if (trace != nullptr && std::strcmp(trace, "1") == 0) {
        std::cerr << "A5_FFT_VARIANT variant=" << (n <= 256 ? "lazy-scatter" : "unrolled-direct")
                  << (doubleBuffer ? "-db" : "-single") << " N=" << n << " effective_batch=" << param.batchSize
                  << " tile_points=" << tilePoints << " aiv_count=" << aivCount << " block_dim=" << blockDim
                  << " DB=" << (doubleBuffer ? 1 : 0) << " workspace_bytes=0\n";
    }
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
