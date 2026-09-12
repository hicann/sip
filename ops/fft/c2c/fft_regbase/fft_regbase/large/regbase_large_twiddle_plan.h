/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_REGBASE_LARGE_TWIDDLE_PLAN_H
#define FFT_REGBASE_LARGE_TWIDDLE_PLAN_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fft::regbase_large {

constexpr std::size_t kLocalFftSize = 128;
constexpr std::size_t kVectorLanesFp32 = 64;
constexpr std::size_t kPhysicalTwiddleStageCount = 6;
constexpr std::size_t kPhysicalTwiddleStageFloats = 2 * kVectorLanesFp32;
constexpr std::size_t kPhysicalTwiddleFloats = kPhysicalTwiddleStageCount * kPhysicalTwiddleStageFloats;

inline std::size_t ReverseLowBits(std::size_t value, std::size_t bitCount)
{
    std::size_t reversed = 0;
    for (std::size_t bit = 0; bit < bitCount; ++bit) {
        reversed = (reversed << 1U) | ((value >> bit) & std::size_t{1});
    }
    return reversed;
}

inline std::vector<float> BuildCompactFft128Twiddles(bool inverse)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<float> compact(2 * kLocalFftSize, 0.0f);
    std::size_t complexOffset = 0;
    for (std::size_t length = 2; length <= kLocalFftSize; length <<= 1U) {
        const std::size_t half = length >> 1U;
        for (std::size_t j = 0; j < half; ++j) {
            const double angle = sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(length);
            compact[2 * (complexOffset + j)] = static_cast<float>(std::cos(angle));
            compact[2 * (complexOffset + j) + 1] = static_cast<float>(std::sin(angle));
        }
        complexOffset += half;
    }
    if (complexOffset != kLocalFftSize - 1) {
        throw std::logic_error("internal FFT128 compact twiddle layout error");
    }
    return compact;
}

// Six SoA planes for the non-trivial Stockham stages half=2..64:
//
//   [stage][64 real coefficients][64 imaginary coefficients]
//
// Each 64-FP32 plane is one 256-byte A5 vector load.  Values are copied from
// the compact production table so the two layouts are bit-identical.
inline std::vector<float> BuildPhysicalFft128Twiddles(bool inverse)
{
    const std::vector<float> compact = BuildCompactFft128Twiddles(inverse);
    std::vector<float> physical(kPhysicalTwiddleFloats, 0.0f);
    for (std::size_t stage = 0; stage < kPhysicalTwiddleStageCount; ++stage) {
        const std::size_t half = std::size_t{2} << stage;
        const std::size_t bitCount = stage + 1;
        const std::size_t stageBase = stage * kPhysicalTwiddleStageFloats;
        for (std::size_t lane = 0; lane < kVectorLanesFp32; ++lane) {
            const std::size_t j = ReverseLowBits(lane, bitCount);
            const std::size_t compactComplex = half - 1 + j;
            physical[stageBase + lane] = compact[2 * compactComplex];
            physical[stageBase + kVectorLanesFp32 + lane] = compact[2 * compactComplex + 1];
        }
    }
    return physical;
}

// FFT256 uses FFT128 of even/odd inputs followed by the final radix-2
// butterfly E[k] +/- W256^k O[k]. Retain the physical FFT128 lane order.
inline std::vector<float> BuildPhysicalFft256Twiddles(bool inverse)
{
    auto table = BuildPhysicalFft128Twiddles(inverse);
    table.resize(kPhysicalTwiddleFloats + 256);
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t lane = 0; lane < 64; ++lane) {
            const auto j = ReverseLowBits(lane, 6) + 64 * half;
            const double angle = (inverse ? 1.0 : -1.0) * 2 * pi * j / 256;
            const auto base = kPhysicalTwiddleFloats + 128 * half;
            table[base + lane] = static_cast<float>(std::cos(angle));
            table[base + 64 + lane] = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

inline std::vector<float> BuildPhysicalFft512Twiddles(bool inverse)
{
    auto table = BuildPhysicalFft256Twiddles(inverse);
    table.resize(1536);
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        for (std::size_t lane = 0; lane < 64; ++lane) {
            const auto j = ReverseLowBits(lane, 6) + 64 * chunk;
            const double angle = (inverse ? 1.0 : -1.0) * 2 * pi * j / 512;
            const auto base = 1024 + 128 * chunk;
            table[base + lane] = static_cast<float>(std::cos(angle));
            table[base + 64 + lane] = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

} // namespace fft::regbase_large

#endif // FFT_REGBASE_LARGE_TWIDDLE_PLAN_H
