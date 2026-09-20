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

#ifndef FFT_TILING_DEF_H
#define FFT_TILING_DEF_H

#include <cstdint>

namespace Fft {

// ---------------------------------------------------------------------------
// Device-side tiling (kernel reads this POD from GM).
//
// The per-core matrix range is NOT stored: the kernel recomputes it from
// (b1, coreNum) with the same FFT89_CORE_MAT_RANGE macro as the host tiling
// (deterministic; test_tiling_host.cpp verifies host/device agreement).
// ---------------------------------------------------------------------------
struct FftTilingData {
    // Input shape
    int64_t b1;        // 12000
    int64_t fftN1;     // 500 (middle dim: column FFT, SIMT)
    int64_t fftN2;     // 50  (last dim: row FFT, GEMM)
    int32_t direction; // -1 forward / +1 inverse (unnormalized)

    // Launch topology
    uint32_t coreNum; // blockDim = AIC count (each block: 1 AIC + 2 AIV)

    // Coefficient byte offsets (within the coeffs GM buffer)
    int64_t twiddleOffset; // this direction's 1040-float SIMT table
    int64_t w50Offset;     // this direction's padded [100,104] GEMM matrix
};

} // namespace Fft

#endif // FFT_TILING_DEF_H
