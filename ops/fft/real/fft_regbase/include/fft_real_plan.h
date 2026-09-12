/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_STANDALONE_REAL_PLAN_H
#define FFT_STANDALONE_REAL_PLAN_H
#include "fft_c2c_plan.h"
#include <cmath>
#include <vector>

namespace fft::real {
enum class Kind { R2C, C2R };
enum class Variant { Auto, Baseline, FusedUb, TiledBoundary };
constexpr int32_t kBoundaryPoints = 256;
constexpr size_t kBoundaryUbBytes = 4 * 2 * kBoundaryPoints * sizeof(float) + 64;
struct Descriptor {
    int32_t n = 0;
    int64_t batch = 0;
    Kind kind = Kind::R2C;
    bool inverse = false;
    Variant variant = Variant::Baseline;
    c2c::Descriptor core;
    uint32_t boundaryBlocks = 0;
    int32_t boundaryTilePoints = kBoundaryPoints;
    size_t boundaryUbBytes = kBoundaryUbBytes;
    size_t inputBytes = 0, outputBytes = 0, scratchBytes = 0;
    size_t workspaceBytes = 0, boundaryTwiddleBytes = 0;
    uint32_t fusedBlocks = 0, fusedTileBatch = 0, fusedUbSlots = 0;
    size_t fusedUbBytes = 0;
    bool IsFused() const { return variant == Variant::FusedUb; }
    int Launches() const { return IsFused() ? 1 : core.Launches() + 1; }
};
inline const char* KindName(Kind kind) { return kind == Kind::R2C ? "r2c" : "c2r"; }
inline const char* VariantName(Variant variant)
{
    return variant == Variant::Auto          ? "auto" :
           variant == Variant::FusedUb       ? "fused-ub" :
           variant == Variant::TiledBoundary ? "tiled-boundary" :
                                               "baseline";
}
inline Descriptor Select(int32_t n, int64_t batch, Kind kind, bool inverse, uint32_t aivCount,
                         Variant variant = Variant::Auto)
{
    // The full-N C2C bound is conservative for the aggregate real buffers too.
    (void)c2c::Select(n, batch, inverse, aivCount);
    if (kind != Kind::R2C && kind != Kind::C2R)
        throw std::invalid_argument("real FFT kind must be R2C or C2R");
    if (variant != Variant::Auto && variant != Variant::Baseline && variant != Variant::FusedUb &&
        variant != Variant::TiledBoundary)
        throw std::invalid_argument("invalid real FFT variant");
    if (variant == Variant::Auto) {
        // Device-validated, forward-measured policy on 56 AIVs. N16384 loses
        // conversion parallelism when fused at small B; B32 is the first
        // measured clear win over both separate-boundary alternatives.
        if (n >= 128 && (n < 16384 || (n == 16384 && batch >= 32)))
            variant = Variant::FusedUb;
        else if (n >= 16384)
            variant = Variant::TiledBoundary;
        else
            variant = Variant::Baseline;
    }
    if (variant == Variant::FusedUb && (n < 128 || n > 16384))
        throw std::invalid_argument("fused-ub experiment supports real N=128..16384 only");
    Descriptor d;
    d.n = n;
    d.batch = batch;
    d.kind = kind;
    d.inverse = inverse;
    const int32_t m = n / 2;
    if (m >= 16) {
        d.core = c2c::Select(m, batch, inverse, aivCount);
    } else {
        // Private N=8 instantiation of the existing packed Stockham body.
        // Public C2C range and all previously measured policies stay unchanged.
        d.core = c2c::Select(16, batch, inverse, aivCount);
        d.core.n = 8;
        d.core.smallVariant = c2c::SmallVariant::FusedUnrolledPackedDirectDoubleBuffer;
        d.core.dataBytes = size_t{64} * batch;
        d.core.twiddleBytes = 64;
        d.core.phaseAUbBytes = 4 * size_t{8} * (256 + 64) + 64;
        d.core.phaseABlocks = static_cast<uint32_t>(std::min<int64_t>(aivCount, c2c::CeilDivide(batch, 16)));
    }
    const size_t realBytes = static_cast<size_t>(batch) * n * sizeof(float);
    const size_t complexBytes = static_cast<size_t>(batch) * (static_cast<size_t>(n) + 2) * sizeof(float);
    d.inputBytes = kind == Kind::R2C ? realBytes : complexBytes;
    d.outputBytes = kind == Kind::R2C ? complexBytes : realBytes;
    d.scratchBytes = realBytes;
    d.workspaceBytes = d.scratchBytes + d.core.workspaceBytes;
    d.boundaryTwiddleBytes = (static_cast<size_t>(m) + 1) * 2 * sizeof(float);
    if (variant == Variant::TiledBoundary) {
        d.variant = variant;
        // Amortize MTE/event setup without reducing the number of occupied
        // AIVs. Keep the original fine-grain boundary available as baseline.
        if (n >= 16384)
            for (int tile : {4096, 1024}) {
                if (batch * c2c::CeilDivide(m - 1, tile) >= aivCount) {
                    d.boundaryTilePoints = tile;
                    break;
                }
            }
        d.boundaryUbBytes = sizeof(float) * 8 * d.boundaryTilePoints + 64;
    }
    const int64_t chunks = c2c::CeilDivide(m - 1, d.boundaryTilePoints);
    d.boundaryBlocks = static_cast<uint32_t>(std::min<int64_t>(aivCount, batch * chunks));
    // Both explicit diagnostic selection and the measured auto policy reach
    // the same retained device bodies. Public C2C dispatch stays unchanged.
    if (variant == Variant::FusedUb) {
        d.variant = variant;
        d.scratchBytes = d.workspaceBytes = 0;
        d.boundaryBlocks = 0;
        d.fusedBlocks = d.core.phaseABlocks;
        d.fusedTileBatch = d.core.tilePoints / m;
        d.fusedUbSlots = n <= 8192 && batch > int64_t{d.fusedBlocks} * d.fusedTileBatch ? 2 : 1;
        // Packed FFT rows N; half-spectrum UB rows N+8 (GM remains N+2).
        // One packed tile, one padded tile, and two coefficient buffers.
        const size_t packedFloats = m >= 512 ? d.fusedTileBatch * (n + 8U) + 128U : d.fusedTileBatch * n;
        const size_t halfFloats = m >= 512 ? packedFloats : d.fusedTileBatch * (n + 8U);
        const size_t realTwFloats = n == 16384 ? n / 2U + 8U : n + 8U;
        d.fusedUbBytes = sizeof(float) * (d.fusedUbSlots * (packedFloats + halfFloats) + n + realTwFloats);
        d.boundaryUbBytes = 0;
        if (d.fusedUbBytes > c2c::kUserUbBytes)
            throw std::logic_error("fused UB budget exceeded");
    }
    return d;
}
inline std::vector<float> BuildBoundaryTwiddles(const Descriptor& d)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<float> table(d.boundaryTwiddleBytes / sizeof(float));
    for (int32_t k = 0; k <= d.n / 2; ++k) {
        const double angle = (d.inverse ? 1.0 : -1.0) * 2.0 * pi * k / d.n;
        table[2 * k] = static_cast<float>(std::cos(angle));
        table[2 * k + 1] = static_cast<float>(std::sin(angle));
    }
    table[0] = 1.0f;
    table[1] = 0.0f;
    table[d.n] = -1.0f;
    table[d.n + 1] = 0.0f;
    return table;
}
} // namespace fft::real
#endif
