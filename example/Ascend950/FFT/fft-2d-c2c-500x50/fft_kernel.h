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

#ifndef FFT_KERNEL_H
#define FFT_KERNEL_H

#include <cstdint>
#include <acl/acl.h>

namespace Fft {

// Launch the fused 2D C2C FFT kernel:
//
//   AIC : per matrix, 2 M-half GEMMs (ex88 row FFT) whose Fixpipe L0C->UB
//         egress lands in the owner AIV's UB tile [500, 104];
//   AIV : per owned matrix, simt_fft_500x50 (ex89 column FFT port) reads
//         the tile and scatters digit-reversed results to GM.
//
// input/output: [b1, fftN1, fftN2] complex64 (interleaved), device buffers.
// coeffs: the BuildAllCoeffs buffer (twiddles fwd/inv + W50 fwd/inv).
// tiling: the FftTilingData POD (filled by FillDeviceTiling).
void FftC2C2DFusedKernelLaunch(GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR tiling, uint32_t blockDim,
                               aclrtStream stream);

} // namespace Fft

#endif // FFT_KERNEL_H
