/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFT_C2C_REGBASE_H
#define ASDSIP_PARAMS_FFT_C2C_REGBASE_H

#include <cstdint>
#include <sstream>
#include <string>

namespace AsdSip {
namespace OpParam {
inline bool IsFftC2CRegBaseSupported(int64_t n)
{
    switch (n) {
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
            return true;
        default:
            return false;
    }
}

// Coefficients are N complex FP32 values in compact Stockham stage order:
// stage half H starts at complex offset H-1. The last value is zero padding.
// The coefficient sign encodes direction; neither the kernel nor tiling
// conjugates or normalizes the result.
struct FftC2CRegBase {
    int64_t fftN = 0;
    int64_t batchSize = 0;
    int32_t isInverse = 0;

    bool operator==(const FftC2CRegBase& other) const
    {
        return fftN == other.fftN && batchSize == other.batchSize && isInverse == other.isInverse;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftC2CRegBase, fftN:" << fftN << ", batchSize:" << batchSize << ", isInverse:" << isInverse;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip
#endif
