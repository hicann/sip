/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "FFTMixCache.h"

namespace op_api {

static FFTMixCache fftCache(FFTMixCache::DEFAULT_CACHE_SIZE);

FFTMixCache::FFTMixCache(int64_t c) noexcept : capacity(c) {}

FFTCacheValue FFTMixCache::get(FFTCacheKey& cacheKey)
{
    std::lock_guard<std::mutex> guard(fftMutex);
    auto match_key = [&cacheKey](FFTPair& pair) { return pair.first == cacheKey; };
    auto it = std::find_if(list.begin(), list.end(), match_key);
    if (it != list.end()) {
        FFTPair tmp = *it;
        list.push_back(tmp);
        list.erase(it);
        return tmp.second;
    }

    if (static_cast<int>(list.size()) >= capacity) {
        destroyFftHandle(list.front().second.handle);
        list.pop_front();
    }

    FFTCacheValue value;
    value.handle = createFftHandle(cacheKey.fftParam);
    if (value.handle == nullptr) {
        // 创建失败：不入缓存，避免坏 handle 固化为持续故障，下次调用可重建（issue #132）
        return value;
    }
    list.push_back(std::make_pair(cacheKey, value));
    return value;
}

void FFTMixCache::setCapacity(int64_t maxSize)
{
    std::lock_guard<std::mutex> guard(fftMutex);
    if (capacity == maxSize) {
        return;
    }

    capacity = maxSize;
    while (static_cast<int>(list.size()) > capacity) {
        destroyFftHandle(list.front().second.handle);
        list.pop_front();
    }
}

int64_t FFTMixCache::getCapacity() { return capacity; }

int64_t FFTMixCache::getSize() { return list.size(); }

void FFTMixCache::clear()
{
    for (auto it = list.begin(); it != list.end(); ++it) {
        FFTPair tmp = *it;
        destroyFftHandle(tmp.second.handle);
    }
    list.clear();
}

asdFftHandle getFftHandle(FFTParam& param)
{
    FFTCacheKey cacheKey;
    cacheKey.fftParam = param;
    return fftCache.get(cacheKey).handle;
}
} // namespace op_api
