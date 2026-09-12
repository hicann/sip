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
#include "fft_real_dispatch.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace AsdSip::OpParam {
struct FftRealStockham {
    int64_t fftN;
    int64_t batchSize;
    int32_t isInverse;
    int32_t isC2R;
    bool operator==(const FftRealStockham& other) const
    {
        return fftN == other.fftN && batchSize == other.batchSize && isInverse == other.isInverse &&
               isC2R == other.isC2R;
    }
    std::string ToString() const
    {
        return "FftRealStockham N=" + std::to_string(fftN) + " batch=" + std::to_string(batchSize) +
               " inverse=" + std::to_string(isInverse) + " c2r=" + std::to_string(isC2R);
    }
};

// Host-only plan and coefficient storage. No input samples are touched here.
struct RealStockhamPlan {
    int64_t complexN;
    size_t inputBytes, outputBytes, bufferBytes;
    size_t dftFloats = 0, twiddleFloats = 0, realFloats;
    std::vector<int32_t> radices;
    explicit RealStockhamPlan(const FftRealStockham& p)
    {
        if (!fft::a5::SupportsRealStockham(p.fftN, p.batchSize) || (p.isInverse != 0 && p.isInverse != 1) ||
            (p.isC2R != 0 && p.isC2R != 1))
            throw std::invalid_argument("Unsupported non-radix-2 real Stockham plan");
        complexN = (p.fftN & 1) ? p.fftN : p.fftN / 2;
        const size_t realBytes = size_t(p.batchSize) * p.fftN * sizeof(float);
        const size_t spectrumBytes = size_t(p.batchSize) * (p.fftN / 2 + 1) * 2 * sizeof(float);
        inputBytes = p.isC2R ? spectrumBytes : realBytes;
        outputBytes = p.isC2R ? realBytes : spectrumBytes;
        bufferBytes = size_t(p.batchSize) * complexN * 2 * sizeof(float);
        // Preserve the donor layouts: C2R uses M twiddles, R2C uses floor(N/2)+1.
        realFloats = size_t(p.isC2R ? complexN : p.fftN / 2 + 1) * 2;
        int64_t remaining = complexN, prev = 1;
        for (int32_t radix : {2, 3, 5, 7}) {
            while (remaining % radix == 0) {
                radices.push_back(radix);
                dftFloats += 2 * radix * radix;
                twiddleFloats += 2 * radix * prev;
                prev *= radix;
                remaining /= radix;
            }
        }
    }
};

inline bool StockhamComplexSignal(const FftRealStockham& p, bool output) { return (p.isC2R != 0) != output; }
inline int64_t StockhamSignalElements(const FftRealStockham& p, bool output)
{
    return StockhamComplexSignal(p, output) ? p.fftN / 2 + 1 : p.fftN;
}
inline const char* StockhamKernelName(bool c2r)
{
    return c2r ? "FftRealStockhamC2RKernel" : "FftRealStockhamR2CKernel";
}
inline std::vector<float> BuildStockhamCoefficients(const FftRealStockham& p, const RealStockhamPlan& plan,
                                                    unsigned role)
{
    constexpr double twoPi = 6.283185307179586476925286766559;
    const double sign = p.isInverse ? 1.0 : -1.0;
    std::vector<float> values;
    auto append = [&](double angle) {
        values.push_back(static_cast<float>(std::cos(angle)));
        values.push_back(static_cast<float>(std::sin(angle)));
    };
    if (role == 1) {
        values.reserve(plan.dftFloats);
        for (int64_t radix : plan.radices)
            for (int64_t q = 0; q < radix; ++q)
                for (int64_t r = 0; r < radix; ++r)
                    append(sign * twoPi * r * q / radix);
    } else if (role == 2) {
        values.reserve(plan.twiddleFloats);
        int64_t prev = 1;
        for (int64_t radix : plan.radices) {
            const int64_t length = prev * radix;
            for (int64_t r = 0; r < radix; ++r)
                for (int64_t j = 0; j < prev; ++j)
                    append(sign * twoPi * r * j / length);
            prev = length;
        }
    } else if (role == 3) {
        values.reserve(plan.realFloats);
        for (size_t k = 0; k < plan.realFloats / 2; ++k)
            append(sign * twoPi * k / p.fftN);
    } else {
        throw std::invalid_argument("Unknown Stockham coefficient role");
    }
    return values;
}
} // namespace AsdSip::OpParam
