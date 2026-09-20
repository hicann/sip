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

#ifndef FFT_COEFFS_H
#define FFT_COEFFS_H

#include <cstdint>
#include "fft_types.h"

namespace Fft {

// ---------------------------------------------------------------------------
// Coefficient builders (host side, one aclrtMemcpy to GM at plan time):
//
// Stage R (AIC GEMM): BuildRotationEmbeddedPadded — ex88's embedded
// rotation matrix [100, 100] zero-padded to [100, 104] row-major (the 4 pad
// columns satisfy the Fixpipe nSize 8-float alignment; the GEMM result in
// those columns is always 0 and the SIMT kernel never reads them).
//
// Stage C (AIV SIMT): BuildTwiddles — ex89's 1040-float interleaved twiddle
// table per direction (radix-4/5 tables; see fft_types.h offsets).
//
// Combined buffer layout (floats):
//   twFwd[1040] | twInv[1040] | Wfwd[100*104] | Winv[100*104]
// ---------------------------------------------------------------------------

// Build one direction's padded rotation matrix [100, 104] into dst
// (row-major).  direction: -1 forward / +1 inverse.
void BuildRotationEmbeddedPadded(float* dst, int32_t N2, int32_t stride, int32_t direction);

// Build one direction's 1040-float twiddle table into dst (ex89 layout).
void BuildTwiddlesOneDirection(float* dst, int32_t direction);

// Build the whole coefficient buffer (GetCoeffsSize bytes).
void BuildAllCoeffs(float* coeffs);

// Total coefficient buffer size in bytes.
size_t GetCoeffsSize();

// Byte offset (within the BuildAllCoeffs buffer) of this direction's W50.
int64_t GetW50Offset(int32_t direction);

// Byte offset of this direction's twiddle table.
int64_t GetTwiddleOffset(int32_t direction);

} // namespace Fft

#endif // FFT_COEFFS_H
