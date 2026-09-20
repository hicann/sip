/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft_tiling.h"

#include "fft_coeffs.h"
#include "fft_tiling_def.h"
#include "fft_types.h"

#include <iostream>

namespace Fft {

bool ComputeTiling(FftHostTilingData& tiling, int64_t b1, int64_t fftN1, int64_t fftN2, int32_t direction,
                   int32_t cubeCoreNum)
{
    tiling.b1 = b1;
    tiling.fftN1 = fftN1;
    tiling.fftN2 = fftN2;
    tiling.direction = direction;
    tiling.cubeCoreNum = static_cast<uint32_t>(cubeCoreNum);

    tiling.matsPerCore = b1 / static_cast<int64_t>(cubeCoreNum);
    tiling.remCores = b1 % static_cast<int64_t>(cubeCoreNum);

    tiling.coeffsSize = GetCoeffsSize();
    tiling.workspaceSize = 0; // fused AIC->UB->AIV pipeline: no GM workspace
    tiling.tilingSize = sizeof(FftTilingData);

    if (fftN1 != FFT_N1_DEFAULT) {
        std::cerr << "FftTiling: fftN1=" << fftN1 << " unsupported (this operator "
                  << "is specialized for fftN1=" << FFT_N1_DEFAULT << " = 4*5*5*5)" << std::endl;
        return false;
    }
    if (fftN2 != FFT_N2_DEFAULT) {
        std::cerr << "FftTiling: fftN2=" << fftN2 << " unsupported (this operator "
                  << "is specialized for fftN2=" << FFT_N2_DEFAULT << " = row-FFT GEMM dim)" << std::endl;
        return false;
    }
    if (cubeCoreNum < 1) {
        std::cerr << "FftTiling: invalid coreNum=" << cubeCoreNum << std::endl;
        return false;
    }
    return true;
}

void FillDeviceTiling(const FftHostTilingData& host, FftTilingData& dev)
{
    dev.b1 = host.b1;
    dev.fftN1 = host.fftN1;
    dev.fftN2 = host.fftN2;
    dev.direction = host.direction;
    dev.coreNum = host.cubeCoreNum;
    dev.twiddleOffset = GetTwiddleOffset(host.direction);
    dev.w50Offset = GetW50Offset(host.direction);
}

} // namespace Fft
