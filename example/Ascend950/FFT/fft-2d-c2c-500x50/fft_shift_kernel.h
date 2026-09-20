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

#ifndef FFT_SHIFT_KERNEL_H
#define FFT_SHIFT_KERNEL_H

#include <cstdint>
#include <acl/acl.h>

namespace Fft {

// ---------------------------------------------------------------------------
// Device-side tiling for the 2D fftshift / ifftshift kernel.
// ---------------------------------------------------------------------------
struct FftShiftTilingData {
    int64_t b1;       // leading batch dim (number of [fftN1, fftN2] slices)
    uint32_t coreNum; // number of AIV cores launched
    int32_t inverse;  // 0 = fftshift (scale 1), 1 = ifftshift (scale 1/N)
};

// ---------------------------------------------------------------------------
// Launch the AIV-only fftshift / ifftshift kernel.
//
//   input : device buffer, [b1, fftN1, fftN2] complex64 (interleaved float).
//   output: device buffer, same layout, cannot alias input (out-of-place).
//   tiling: FftShiftTilingData POD already copied to GM.
//   blockDim: number of AIV cores to launch (coreNum of the tiling).
// ---------------------------------------------------------------------------
void FftShift2DKernelLaunch(GM_ADDR input, GM_ADDR output, GM_ADDR tiling, uint32_t blockDim, aclrtStream stream);

} // namespace Fft

#endif // FFT_SHIFT_KERNEL_H
