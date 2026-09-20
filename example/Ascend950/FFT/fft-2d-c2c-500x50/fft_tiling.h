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

#ifndef FFT_TILING_H
#define FFT_TILING_H

#include <cstdint>

namespace Fft {

// Host-side mirror of the launch decisions (informational + unit-testable).
struct FftHostTilingData {
    int64_t b1{0};
    int64_t fftN1{0};
    int64_t fftN2{0};
    int32_t direction{-1};
    uint32_t cubeCoreNum{0};

    // Balanced split summary
    int64_t matsPerCore{0}; // b1 / coreNum
    int64_t remCores{0};    // b1 % coreNum (these take one extra matrix)

    size_t coeffsSize{0};
    size_t workspaceSize{0}; // always 0: fused in-UB pipeline, no GM workspace
    size_t tilingSize{0};
};

// Validates the shape (fftN1 == 500, fftN2 == 50 — the kernel is
// specialized for this) and fills the host tiling.  Returns false on shape
// mismatch.
bool ComputeTiling(FftHostTilingData& tiling, int64_t b1, int64_t fftN1, int64_t fftN2, int32_t direction,
                   int32_t cubeCoreNum);

// Fill the device POD from the host tiling.
void FillDeviceTiling(const FftHostTilingData& host, struct FftTilingData& dev);

} // namespace Fft

#endif // FFT_TILING_H
