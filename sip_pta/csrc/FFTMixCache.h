/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFTMIXCACHE_H
#define FFTMIXCACHE_H

#include <utility>
#include <list>
#include <mutex>
#include <algorithm>
#include <cstddef>

#include "fft_api.h"

using namespace AsdSip;

namespace op_api {
struct FFTParam {
    int64_t fftXSize = 0;
    int64_t fftYSize = 0;
    int64_t fftZSize = 0;
    asdFftType fftType = asdFftType::ASCEND_FFT_C2C;
    asdFftDirection direction = asdFftDirection::ASCEND_FFT_FORWARD;
    int64_t batchSize = 0;
    asdFft1dDimType dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
};
inline bool operator==(const FFTParam& one, const FFTParam& other)
{
    return one.fftXSize == other.fftXSize && one.fftYSize == other.fftYSize && one.fftZSize == other.fftZSize &&
           one.fftType == other.fftType && one.direction == other.direction && one.batchSize == other.batchSize &&
           one.dimType == other.dimType;
}
// For torch_npu._C
asdFftHandle getFftHandle(FFTParam& param);

struct FFTCacheKey {
    FFTParam fftParam;
};
struct FFTCacheValue {
    asdFftHandle handle = nullptr;
};

inline bool operator==(const FFTCacheKey& one, const FFTCacheKey& other) { return one.fftParam == other.fftParam; }

class FFTMixCache {
public:
    static constexpr int DEFAULT_CACHE_SIZE = 10;
    using FFTPair = std::pair<FFTCacheKey, FFTCacheValue>;
    explicit FFTMixCache(int64_t c) noexcept;
    FFTCacheValue get(FFTCacheKey& cacheKey);
    void setCapacity(int64_t maxSize);
    int64_t getCapacity();
    int64_t getSize();
    void clear();

private:
    int64_t capacity;
    std::list<FFTPair> list{};
    std::mutex fftMutex;
};

inline asdFftHandle createFftHandle(const FFTParam& param)
{
    asdFftHandle handle = nullptr;
    if (asdFftCreate(handle) != AsdSip::ErrorType::ACL_SUCCESS) {
        return nullptr;
    }
    AspbStatus planStatus = AsdSip::ErrorType::ACL_SUCCESS;
    if (param.fftYSize == 0 && param.fftZSize == 0) {
        planStatus = asdFftMakePlan1D(handle, param.fftXSize, param.fftType, param.direction, param.batchSize,
                                      param.dimType);
    } else if (param.fftZSize == 0) {
        planStatus = asdFftMakePlan2D(handle, param.fftXSize, param.fftYSize, param.fftType, param.direction,
                                      param.batchSize);
    } else {
        planStatus = asdFftMakePlan3D(handle, param.fftXSize, param.fftYSize, param.fftZSize, param.fftType,
                                      param.direction, param.batchSize);
    }
    if (planStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        // plan 创建失败：销毁句柄并返回空，本次调用失败但下次调用可重建（issue #132）
        asdFftDestroy(handle);
        return nullptr;
    }
    return handle;
}
inline void destroyFftHandle(asdFftHandle handle)
{
    asdFftSynchronize(handle);
    asdFftDestroy(handle);
}
} // namespace op_api
#endif
