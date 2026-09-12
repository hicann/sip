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
#include <sstream>
#include <string>
#include "fft_c2c_dispatch.h"

namespace AsdSip::OpParam {
enum class FftLargeStage : int32_t {
    RegA = 1,
    RegB = 2,
    OuterRegister = 3,
    OuterSimd = 4,
    OuterRadix4 = 5,
    OuterRadix8 = 6,
    CachedA = 7,
    ExtendedA = 8,
    ExtendedB = 9,
    Packed = 10,
    Gauss = 11
};

struct FftC2CLargeStage {
    int64_t fftN = 0;
    int64_t batchSize = 0;
    int32_t isInverse = 0;
    FftLargeStage stage = FftLargeStage::RegA;

    bool operator==(const FftC2CLargeStage& other) const
    {
        return fftN == other.fftN && batchSize == other.batchSize && isInverse == other.isInverse &&
               stage == other.stage;
    }
    std::string ToString() const
    {
        std::ostringstream s;
        s << "OpName: FftC2CLargeStage, fftN:" << fftN << ", batchSize:" << batchSize << ", inverse:" << isInverse
          << ", stage:" << static_cast<int32_t>(stage);
        return s.str();
    }
};

inline bool IsFftLargeStageSupported(const FftC2CLargeStage& p)
{
    const auto n = p.fftN;
    if (n < 16384 || n > 1048576 || (n & (n - 1)) || p.batchSize < 1 || (p.isInverse != 0 && p.isInverse != 1))
        return false;
    // Reserve up to three signal allocations and coefficient/alignment space.
    const uint64_t maxBytes = (std::numeric_limits<size_t>::max() - 32 * 1024 * 1024) / 3;
    if (static_cast<uint64_t>(p.batchSize) > maxBytes / (8 * n))
        return false;
    using S = FftLargeStage;
    if (p.stage == S::RegA || p.stage == S::RegB)
        return n <= 131072;
    if (!fft::a5::FitsCubeRowIndex(n, p.batchSize))
        return false;
    switch (p.stage) {
        case S::OuterRegister:
        case S::ExtendedA:
        case S::ExtendedB:
            return n == 65536 || n == 131072;
        case S::OuterSimd:
            return n == 262144;
        case S::OuterRadix4:
            return n == 524288;
        case S::OuterRadix8:
            return n == 1048576;
        case S::CachedA:
            return n == 16384 || n >= 262144;
        case S::Packed:
            return n == 16384 || n == 262144;
        case S::Gauss:
            return n >= 262144;
        default:
            return false;
    }
}

inline const char* FftLargeKernelName(FftLargeStage stage)
{
    using S = FftLargeStage;
    switch (stage) {
        case S::RegA:
            return "FftC2CRegBaseLargeAKernel";
        case S::RegB:
            return "FftC2CRegBaseLargeBKernel";
        case S::OuterRegister:
        case S::OuterSimd:
        case S::OuterRadix4:
        case S::OuterRadix8:
            return "FftC2COuterKernel";
        case S::CachedA:
        case S::ExtendedA:
            return "FftC2CCubeVectorAKernel";
        case S::ExtendedB:
            return "FftC2CCubeVectorBKernel";
        case S::Packed:
            return "FftC2CPackedKernel";
        case S::Gauss:
            return "FftC2CGaussKernel";
    }
    return "";
}

inline int64_t FftLargeCoefficientCount(const FftC2CLargeStage& p, int index)
{
    using S = FftLargeStage;
    if (index == 2) {
        switch (p.stage) {
            case S::RegA:
            case S::OuterRegister:
            case S::OuterSimd:
            case S::OuterRadix4:
            case S::OuterRadix8:
                return 2 * p.fftN;
            case S::CachedA:
            case S::ExtendedA:
                return 32768;
            default:
                return 1; // Unused argument retains a valid GM pointer.
        }
    }
    switch (p.stage) {
        case S::RegA:
        case S::RegB:
            return p.fftN == 16384 ? 768 : p.fftN <= 65536 ? 1024 : 1536;
        case S::OuterRegister:
        case S::OuterSimd:
        case S::OuterRadix4:
        case S::OuterRadix8:
            return 128;
        case S::Packed:
            return 4352;
        case S::Gauss:
            return 3328;
        default:
            return 256;
    }
}
} // namespace AsdSip::OpParam
