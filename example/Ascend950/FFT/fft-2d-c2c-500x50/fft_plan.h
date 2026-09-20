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

#ifndef FFT_PLAN_H
#define FFT_PLAN_H

#include <acl/acl.h>
#include <cstdint>

#include "fft_tiling.h"

namespace Fft {

struct FftPlan {
    // Shape
    int64_t b1{0};
    int64_t fftN1{0};
    int64_t fftN2{0};
    int32_t direction{-1};
    int32_t deviceId{0};

    // Host tiling mirror
    FftHostTilingData tiling{};

    // Device buffers
    uint8_t* dCoeffs{nullptr};
    size_t coeffsSize{0};
    uint8_t* dTiling{nullptr};
    size_t tilingSize{0};
};

// Build host coefficients (twiddles both directions + W50 both directions),
// allocate device coeffs + tiling, and copy them over.  Returns false on
// failure.
bool FftPlanCreate(FftPlan& plan, int64_t b1, int64_t fftN1, int64_t fftN2, int32_t direction, int32_t deviceId,
                   int32_t coreNum);

// Launch the fused kernel on the given stream (synchronous wrapper).
void FftExec2D(FftPlan& plan, uint8_t* dInput, uint8_t* dOutput, aclrtStream stream, int32_t coreNum);

// Free device buffers.
void FftPlanDestroy(FftPlan& plan);

} // namespace Fft

#endif // FFT_PLAN_H
