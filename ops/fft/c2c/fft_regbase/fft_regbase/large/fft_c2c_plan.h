/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_STANDALONE_C2C_PLAN_H
#define FFT_STANDALONE_C2C_PLAN_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Only retained compute families. Values preserve experiment provenance.
enum class FftC2CRegBaseSmallVariant : int32_t {
    FusedUnrolledPackedDirectDoubleBuffer = 9,
    FusedRegisterDirect = 10,
    FusedRegisterDirectDoubleBuffer = 11,
    FusedRegisterLazyDirectDoubleBuffer = 13,
    FusedRegisterLazyContiguousDirectDoubleBuffer = 15,
    FusedRadix4UnrolledDirect = 16,
    FusedRadix4UnrolledDirectDoubleBuffer = 17,
};
struct alignas(8) FftC2CRegBaseSmallTilingData {
    int64_t batch;
    int32_t n;
    int32_t tilePoints;
};
struct alignas(8) FftC2CRegBaseLargeTilingData {
    int64_t batch;
    int32_t n;
    int32_t groupSize;
};
static_assert(sizeof(FftC2CRegBaseSmallTilingData) == 16);
static_assert(sizeof(FftC2CRegBaseLargeTilingData) == 16);

namespace fft::c2c {
using SmallVariant = FftC2CRegBaseSmallVariant;
constexpr uint64_t kUserUbBytes = 253952; // tested Ascend950PR_9599 configuration

inline int64_t CeilDivide(int64_t value, int64_t divisor) { return value / divisor + (value % divisor != 0); }
inline bool IsSupportedSize(int32_t n) { return n >= 16 && n <= 131072 && (n & (n - 1)) == 0; }
struct Descriptor {
    int32_t n = 0;
    int64_t batch = 0;
    bool inverse = false;
    uint32_t aivCount = 0;
    int32_t tilePoints = 0, coreGrainPoints = 0;
    int32_t p = 0, q = 0, phaseAGroup = 0, phaseBGroup = 0;
    uint32_t phaseABlocks = 0, phaseBBlocks = 0;
    SmallVariant smallVariant = SmallVariant::FusedRegisterDirectDoubleBuffer;
    size_t dataBytes = 0, workspaceBytes = 0, twiddleBytes = 0, crossBytes = 0;
    size_t phaseAUbBytes = 0, phaseBUbBytes = 0;
    bool Large() const { return n >= 16384; }
    int Launches() const { return Large() ? 2 : 1; }
};
inline const char* VariantName(const Descriptor& d)
{
    if (d.Large()) {
        switch (d.n) {
            case 16384:
                return "factorized-128x128-phase-ab-db-physical-twiddle";
            case 32768:
                return "factorized-128x256-physical-twiddle";
            case 65536:
                return "factorized-256x256-physical-twiddle";
            default:
                return "factorized-256x512-physical-twiddle";
        }
    }
    switch (d.smallVariant) {
        case SmallVariant::FusedUnrolledPackedDirectDoubleBuffer:
            return "fused-unrolled-packed-direct-db";
        case SmallVariant::FusedRegisterDirect:
            return "fused-register-direct";
        case SmallVariant::FusedRegisterDirectDoubleBuffer:
            return "fused-register-direct-db";
        case SmallVariant::FusedRegisterLazyDirectDoubleBuffer:
            return "fused-register-lazy-direct-db";
        case SmallVariant::FusedRegisterLazyContiguousDirectDoubleBuffer:
            return "fused-register-lazy-contiguous-direct-db";
        case SmallVariant::FusedRadix4UnrolledDirect:
            return "fused-radix4-unrolled-direct";
        case SmallVariant::FusedRadix4UnrolledDirectDoubleBuffer:
            return "fused-radix4-unrolled-direct-db";
    }
    throw std::logic_error("invalid selected FFT variant");
}
inline Descriptor Select(int32_t n, int64_t batch, bool inverse, uint32_t aivCount)
{
    if (!IsSupportedSize(n))
        throw std::invalid_argument("FFT size must be a power of two in [16, 131072]");
    if (batch <= 0)
        throw std::invalid_argument("batch must be positive");
    if (!aivCount)
        throw std::invalid_argument("AIV count must be positive");
    // Bound signed device byte offsets AND aggregate resident allocations.
    const uint64_t bytesPerTransform = uint64_t{8} * n;
    const uint64_t maxBytes = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                                                 (std::numeric_limits<size_t>::max() - uint64_t{2} * 1024 * 1024) / 3);
    if (static_cast<uint64_t>(batch) > maxBytes / bytesPerTransform)
        throw std::overflow_error("FFT batch exceeds checked address/allocation range");
    Descriptor d;
    d.n = n;
    d.batch = batch;
    d.inverse = inverse;
    d.aivCount = aivCount;
    d.dataBytes = static_cast<size_t>(batch) * bytesPerTransform;
    if (!d.Large()) {
        d.coreGrainPoints = n <= 256 ? 128 : 1024;
        d.tilePoints = n <= 256 ? 256 : std::max(1024, n);
        if (n <= 64) {
            if (batch == 1)
                d.smallVariant = SmallVariant::FusedUnrolledPackedDirectDoubleBuffer;
            else if (n == 32 && batch <= 64)
                d.smallVariant = SmallVariant::FusedRegisterDirect;
            else
                d.smallVariant = SmallVariant::FusedRegisterDirectDoubleBuffer;
        } else if (n <= 256) {
            // No measured B64..B4096 crossover: switch at the first measured
            // throughput batch instead of claiming an optimized threshold.
            d.smallVariant = batch < 4096 ? SmallVariant::FusedRegisterLazyDirectDoubleBuffer :
                                            SmallVariant::FusedRegisterLazyContiguousDirectDoubleBuffer;
            if (n == 256 && batch >= 4096)
                d.tilePoints = 1024;
        } else {
            const int64_t oneWave = int64_t{aivCount} * (d.tilePoints / n);
            d.smallVariant = n <= 4096 && batch > oneWave ? SmallVariant::FusedRadix4UnrolledDirectDoubleBuffer :
                                                            SmallVariant::FusedRadix4UnrolledDirect;
        }
        const int64_t grain = std::max<int64_t>(1, CeilDivide(d.coreGrainPoints, n));
        d.phaseABlocks = static_cast<uint32_t>(std::min<int64_t>(aivCount, CeilDivide(batch, grain)));
        const bool db = d.smallVariant != SmallVariant::FusedRegisterDirect &&
                        d.smallVariant != SmallVariant::FusedRadix4UnrolledDirect;
        const bool resident = n <= 256 && d.smallVariant != SmallVariant::FusedUnrolledPackedDirectDoubleBuffer;
        d.twiddleBytes = size_t{8} * n;
        d.phaseAUbBytes = (db ? 2 : 1) * (resident ? 1 : 2) * size_t{8} * (d.tilePoints + (resident ? 0 : 64)) +
                          d.twiddleBytes;
    } else {
        d.p = n <= 32768 ? 128 : 256;
        d.q = n / d.p;
        d.phaseAGroup = n == 131072 && batch == 1 ? 16 : 4;
        d.phaseBGroup = n == 16384 ? 16 : 4;
        d.phaseABlocks = static_cast<uint32_t>(std::min<int64_t>(aivCount, batch * (d.q / d.phaseAGroup)));
        d.phaseBBlocks = static_cast<uint32_t>(std::min<int64_t>(aivCount, batch * (d.p / d.phaseBGroup)));
        d.workspaceBytes = d.dataBytes;
        d.twiddleBytes = d.q == 128 ? 3072 : d.q == 256 ? 4096 : 6144;
        d.crossBytes = bytesPerTransform;
        d.phaseAUbBytes = size_t{32} * d.p * d.phaseAGroup + (d.p == 128 ? 3072 : 4096);
        d.phaseBUbBytes = size_t{32} * d.q * d.phaseBGroup + d.twiddleBytes;
    }
    if (std::max(d.phaseAUbBytes, d.phaseBUbBytes) > kUserUbBytes)
        throw std::logic_error("selected FFT exceeds supported user UB budget");
    return d;
}
} // namespace fft::c2c
#endif
