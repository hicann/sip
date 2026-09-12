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
#include <initializer_list>
#include <limits>

namespace fft::a5 {
enum class RealRoute { Existing, Dft, Regbase, Stockham };
inline bool SupportsRealStockham(int64_t n, int64_t batch)
{
    // Upgrade only the existing large, non-radix-2 real fallback. Stockham
    // indexes interleaved complex elements with signed 32-bit integers.
    if (n <= 1024 || batch < 1 || (n & (n - 1)) == 0)
        return false;
    int64_t m = (n & 1) ? n : n / 2;
    if (m > std::numeric_limits<int32_t>::max() / 2 || batch > std::numeric_limits<int32_t>::max() / (2 * m))
        return false;
    for (int64_t radix : {2, 3, 5, 7}) {
        while (m % radix == 0)
            m /= radix;
    }
    return m == 1;
}
// Shared by 1D and the contiguous real pass of 2D/3D. Batch is the number
// of actual real rows, not the public multidimensional batch count.
inline RealRoute SelectReal(int64_t n, int64_t batch, bool c2r)
{
    if (n < 1 || batch < 1)
        return RealRoute::Existing;
    if (n >= 16 && n <= 131072 && (n & (n - 1)) == 0) {
        // Device-2 complete-transform comparisons retain RegBase at B1/B64.
        // Separate-boundary real N16..64 loses to DFT at B128. R2C N32/B128
        // is a practical tie: retain RegBase through that measured point.
        // The B256 exception is qualified by the final handoff script before
        // release; intervening batches are interpolation, not a tuned optimum.
        const int64_t dftBatch = n == 32 && !c2r ? 256 : 128;
        if (n <= 64 && batch >= dftBatch)
            return RealRoute::Dft;
        return RealRoute::Regbase;
    }
    if (n <= 1024)
        return RealRoute::Dft;
    return SupportsRealStockham(n, batch) ? RealRoute::Stockham : RealRoute::Existing;
}
} // namespace fft::a5
