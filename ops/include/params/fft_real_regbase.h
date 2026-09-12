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
#include <sstream>
#include "fft_real_plan.h"

namespace AsdSip::OpParam {
enum class FftRealStage : int32_t { Fused, Boundary, Small };
struct FftRealRegBase {
    int32_t fftN;
    int64_t batchSize;
    int32_t isInverse;
    int32_t isC2R;
    FftRealStage stage;
    bool operator==(const FftRealRegBase& p) const
    {
        return fftN == p.fftN && batchSize == p.batchSize && isInverse == p.isInverse && isC2R == p.isC2R &&
               stage == p.stage;
    }
    std::string ToString() const
    {
        std::ostringstream text;
        text << "FftRealRegBase N=" << fftN << " B=" << batchSize << " inverse=" << isInverse << " c2r=" << isC2R
             << " stage=" << static_cast<int>(stage);
        return text.str();
    }
};
inline fft::real::Descriptor RealDescriptor(const FftRealRegBase& p, uint32_t cores)
{
    if ((p.isInverse != 0 && p.isInverse != 1) || (p.isC2R != 0 && p.isC2R != 1))
        throw std::invalid_argument("Invalid real FFT kind or direction");
    return fft::real::Select(p.fftN, p.batchSize, p.isC2R ? fft::real::Kind::C2R : fft::real::Kind::R2C,
                             p.isInverse != 0, cores);
}
inline const char* RealKernelName(FftRealStage stage)
{
    switch (stage) {
        case FftRealStage::Fused:
            return "FftRealRegBaseFusedKernel";
        case FftRealStage::Boundary:
            return "FftRealRegBaseBoundaryKernel";
        case FftRealStage::Small:
            return "FftRealRegBaseSmallKernel";
    }
    return "";
}
inline int64_t RealSignalFloats(const FftRealRegBase& p, bool output)
{
    if (p.stage == FftRealStage::Small)
        return p.fftN;
    if (p.stage == FftRealStage::Boundary)
        return p.fftN + ((output != bool(p.isC2R)) ? 2 : 0);
    return p.fftN + ((output != bool(p.isC2R)) ? 2 : 0);
}
inline int64_t RealCoefficientFloats(const FftRealRegBase& p, const fft::real::Descriptor& d, int index)
{
    if (index == 1)
        return p.stage == FftRealStage::Boundary ? 1 : d.core.twiddleBytes / sizeof(float);
    return p.stage == FftRealStage::Small ? 1 : d.boundaryTwiddleBytes / sizeof(float);
}
inline uint64_t RealTilingKey(const FftRealRegBase& p, const fft::real::Descriptor& d)
{
    if (p.stage == FftRealStage::Boundary)
        return 2 * d.boundaryTilePoints + p.isC2R;
    if (p.stage == FftRealStage::Small)
        return 64 * d.core.n + 2 * static_cast<int>(d.core.smallVariant) + (d.core.n >= 512 ? p.isInverse : 0);
    const bool contiguous = d.core.smallVariant ==
                            fft::c2c::SmallVariant::FusedRegisterLazyContiguousDirectDoubleBuffer;
    return 16 * p.fftN + 8 * p.isC2R + (p.fftN >= 1024 ? 4 * p.isInverse : 0) + 2 * contiguous +
           (d.fusedUbSlots == 2 ? 1 : 0);
}
} // namespace AsdSip::OpParam
