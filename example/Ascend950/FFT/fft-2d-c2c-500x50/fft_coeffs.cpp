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

#include "fft_coeffs.h"

#include "fft_types.h"

#include <cmath>
#include <cstring>

namespace Fft {

// ---------------------------------------------------------------------------
// Stage R: ex88 BuildRotationEmbedded padded to `stride` columns.
// Row-major [2N, stride]; the 2x2 block at (2p, 2q) encodes the complex
// twiddle exp(direction * i * 2*pi*p*q/N); columns [2N, stride) are zero.
// ---------------------------------------------------------------------------
void BuildRotationEmbeddedPadded(float* dst, int32_t N2, int32_t stride, int32_t direction)
{
    const int32_t dim = 2 * N2;
    std::memset(dst, 0, static_cast<size_t>(dim) * stride * sizeof(float));
    const double sign = static_cast<double>(direction);

    for (int32_t p = 0; p < N2; ++p) {
        for (int32_t q = 0; q < N2; ++q) {
            const double angle = FFT_2PI * static_cast<double>(p * q) / static_cast<double>(N2);
            const double cosVal = std::cos(angle);
            const double sinVal = std::sin(angle);

            dst[(2 * p) * stride + (2 * q)] = static_cast<float>(cosVal);
            dst[(2 * p) * stride + (2 * q + 1)] = static_cast<float>(sign * sinVal);
            dst[(2 * p + 1) * stride + (2 * q)] = static_cast<float>(-sign * sinVal);
            dst[(2 * p + 1) * stride + (2 * q + 1)] = static_cast<float>(cosVal);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage C: ex89 BuildOneDirection verbatim (1040-float twiddle table).
// ---------------------------------------------------------------------------
void BuildTwiddlesOneDirection(float* dst, int32_t direction)
{
    const double sign = static_cast<double>(direction);

    auto put = [dst](int32_t floatOff, int32_t cidx, double angle) {
        dst[floatOff + 2 * cidx] = static_cast<float>(std::cos(angle));
        dst[floatOff + 2 * cidx + 1] = static_cast<float>(std::sin(angle));
    };

    for (int32_t j = 0; j < R4_T1_COUNT; ++j) {
        put(R4_T1_OFF, j, sign * FFT_2PI * static_cast<double>(j) / 500.0);
    }
    for (int32_t j = 0; j < R4_T2_COUNT; ++j) {
        put(R4_T2_OFF, j, sign * FFT_2PI * static_cast<double>(j) / 250.0);
    }
    for (int32_t j = 0; j < R4_T3_COUNT; ++j) {
        put(R4_T3_OFF, j, sign * FFT_2PI * static_cast<double>(3 * j) / 500.0);
    }
    for (int32_t j = 0; j < TW3_COUNT / 4; ++j) {
        for (int32_t t = 1; t < 5; ++t) {
            put(TW3_OFF, j * 4 + (t - 1), sign * FFT_2PI * static_cast<double>(j * t) / 125.0);
        }
    }
    for (int32_t j = 0; j < TW4_COUNT / 4; ++j) {
        for (int32_t t = 1; t < 5; ++t) {
            put(TW4_OFF, j * 4 + (t - 1), sign * FFT_2PI * static_cast<double>(j * t) / 25.0);
        }
    }
    for (int32_t s = 0; s < 5; ++s) {
        for (int32_t t = 0; t < 5; ++t) {
            put(W5_OFF, s * 5 + t, sign * FFT_2PI * static_cast<double>(s * t) / 5.0);
        }
    }
}

size_t GetCoeffsSize()
{
    const int64_t total = TW_TOTAL_FLOATS + 2 * W_FLOATS_PER_DIR;
    return static_cast<size_t>(total) * sizeof(float);
}

void BuildAllCoeffs(float* coeffs)
{
    std::memset(coeffs, 0, GetCoeffsSize());
    BuildTwiddlesOneDirection(coeffs, -1);                       // fwd
    BuildTwiddlesOneDirection(coeffs + COEFF_FLOATS_PER_DIR, 1); // inv
    BuildRotationEmbeddedPadded(coeffs + W_FWD_OFF, FFT_N2_DEFAULT, WS_STRIDE, -1);
    BuildRotationEmbeddedPadded(coeffs + W_INV_OFF, FFT_N2_DEFAULT, WS_STRIDE, 1);
}

int64_t GetW50Offset(int32_t direction)
{
    return (direction == 1) ? W_INV_OFF * static_cast<int64_t>(sizeof(float)) :
                              W_FWD_OFF * static_cast<int64_t>(sizeof(float));
}

int64_t GetTwiddleOffset(int32_t direction)
{
    return (direction == 1) ? static_cast<int64_t>(COEFF_FLOATS_PER_DIR) * sizeof(float) : 0;
}

} // namespace Fft
