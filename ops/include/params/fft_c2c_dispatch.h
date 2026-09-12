/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
#include <limits>

// Proposed common SiP C2C policy. Select using each pass's actual length and
// effective batch. Existing means retain the established mixed-radix/SIMT path.
// The measurement record explains the 5% tie margin and unmeasured intervals.
namespace fft::a5 {
enum class Route { Existing, Dft, Regbase, VectorExtended, Packed16, PackedSimd, GaussSimd, GaussRadix4, GaussRadix8 };
inline bool FitsCubeRowIndex(int64_t n, int64_t batch)
{
    if (n < 16384 || n > 1048576 || (n & (n - 1)) || batch < 1)
        return false;
    // rows=128*batch*(n/16384), rounded to a 32-row tile. GM offsets are int64.
    // This also bounds the separate outer stage's batch*columnTileCount.
    return batch <= (std::numeric_limits<int32_t>::max() - 31) / (128 * (n / 16384));
}
inline Route Select(int64_t n, int64_t batch)
{
    if (n < 1 || batch < 1 || (n & (n - 1)))
        return Route::Existing;
    if (n < 128)
        return Route::Dft;
    if (n <= 8192)
        return Route::Regbase;
    switch (n) {
        case 16384:
            // B513 forward was a practical tie: prefer RegBase. B512 is a
            // conservative rounded boundary, not a claimed exact crossover.
            return batch >= 64 && batch < 512 ? Route::Packed16 : Route::Regbase;
        case 32768:
            return Route::Regbase;
        case 65536:
        case 131072:
            return batch == 1 ? Route::VectorExtended : Route::Regbase;
        case 262144:
            if (!FitsCubeRowIndex(n, batch))
                return Route::Existing;
            return batch < 64 ? Route::PackedSimd : Route::GaussSimd;
        case 524288:
            return FitsCubeRowIndex(n, batch) ? Route::GaussRadix4 : Route::Existing;
        case 1048576:
            return FitsCubeRowIndex(n, batch) ? Route::GaussRadix8 : Route::Existing;
        default:
            return Route::Existing;
    }
}
inline const char* Name(Route route)
{
    switch (route) {
        case Route::Existing:
            return "existing";
        case Route::Dft:
            return "dft";
        case Route::Regbase:
            return "regbase_new";
        case Route::VectorExtended:
            return "regbase_extended";
        case Route::Packed16:
            return "packed16";
        case Route::PackedSimd:
            return "packed_simd";
        case Route::GaussSimd:
            return "gauss_simd";
        case Route::GaussRadix4:
            return "gauss_radix4";
        case Route::GaussRadix8:
            return "gauss_radix8";
    }
    return "existing";
}
} // namespace fft::a5
