/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_STANDALONE_C2C_COEFFICIENTS_H
#define FFT_STANDALONE_C2C_COEFFICIENTS_H
#include "fft_c2c_plan.h"
#include "regbase_large_twiddle_plan.h"
namespace fft::c2c {
// Production table builder, tested directly by dispatch_tests.
// Preserve the experiment's compact R2 vs compact R4+R2-tail distinction.
inline std::vector<float> BuildLocalTwiddles(const Descriptor& d)
{
    if (d.Large()) {
        if (d.q == 128)
            return regbase_large::BuildPhysicalFft128Twiddles(d.inverse);
        if (d.q == 256)
            return regbase_large::BuildPhysicalFft256Twiddles(d.inverse);
        return regbase_large::BuildPhysicalFft512Twiddles(d.inverse);
    }
    constexpr double kPi = 3.141592653589793238462643383279502884;
    std::vector<float> table;
    // N complex values are allocated and copied. Only N-1 are live; the last
    // complex value remains zero padding for an aligned MTE2 transfer.
    table.assign(static_cast<size_t>(d.n) * 2, 0.0f);
    const double sign = d.inverse ? 1.0 : -1.0;
    size_t liveTwiddles = 0;
    if (d.n >= 512) {
        int32_t quarter = 1;
        int32_t stages = 0;
        for (int32_t value = d.n; value > 1; value >>= 1)
            ++stages;
        for (int32_t pass = 0; pass < stages / 2; ++pass) {
            const size_t base = static_cast<size_t>(quarter - 1);
            if (base != liveTwiddles) {
                throw std::logic_error("internal radix-4 compact twiddle offset error");
            }
            const int32_t length = 4 * quarter;
            for (int32_t power = 1; power <= 3; ++power) {
                for (int32_t j = 0; j < quarter; ++j) {
                    const double angle = sign * 2.0 * kPi * static_cast<double>(power * j) /
                                         static_cast<double>(length);
                    const size_t complexIndex = base + static_cast<size_t>((power - 1) * quarter + j);
                    const size_t index = complexIndex * 2;
                    table[index] = static_cast<float>(std::cos(angle));
                    table[index + 1] = static_cast<float>(std::sin(angle));
                }
            }
            liveTwiddles += static_cast<size_t>(3 * quarter);
            quarter *= 4;
        }
        if ((stages & 1) != 0) {
            // The last unfused Stockham stage keeps the radix-2 layout used by
            // StockhamRadix2StageBody: base=half-1, length=N.
            const int32_t half = d.n / 2;
            const size_t base = static_cast<size_t>(half - 1);
            if (base != liveTwiddles) {
                throw std::logic_error("internal mixed radix-4/radix-2 twiddle offset error");
            }
            for (int32_t j = 0; j < half; ++j) {
                const double angle = sign * 2.0 * kPi * static_cast<double>(j) / static_cast<double>(d.n);
                const size_t index = (base + static_cast<size_t>(j)) * 2;
                table[index] = static_cast<float>(std::cos(angle));
                table[index + 1] = static_cast<float>(std::sin(angle));
            }
            liveTwiddles += static_cast<size_t>(half);
        }
    } else {
        for (int32_t length = 2; length <= d.n; length <<= 1) {
            const int32_t half = length >> 1;
            for (int32_t j = 0; j < half; ++j) {
                const double angle = sign * 2.0 * kPi * static_cast<double>(j) / static_cast<double>(length);
                const size_t index = (liveTwiddles + static_cast<size_t>(j)) * 2;
                table[index] = static_cast<float>(std::cos(angle));
                table[index + 1] = static_cast<float>(std::sin(angle));
            }
            liveTwiddles += static_cast<size_t>(half);
        }
    }
    if (liveTwiddles != static_cast<size_t>(d.n - 1)) {
        throw std::logic_error("internal compact twiddle layout error");
    }
    return table;
}
} // namespace fft::c2c
#endif
