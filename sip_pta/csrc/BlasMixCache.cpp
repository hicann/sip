/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "BlasMixCache.h"

namespace op_api {

// Cache 容量默认设为 16
static BlasMixCache blasCache(16);

BlasMixCache::BlasMixCache(int64_t c) noexcept : capacity(c) {}

BlasCacheValue BlasMixCache::get(BlasCacheKey& cacheKey, std::function<AspbStatus(asdBlasHandle)> makePlanFunc)
{
    std::lock_guard<std::mutex> guard(blasMutex);

    // 1. 查找缓存
    auto it = std::find_if(list.begin(), list.end(), [&](BlasPair& pair) { return pair.first == cacheKey; });
    if (it != list.end()) {
        BlasPair tmp = *it;
        list.push_back(tmp);
        list.erase(it);
        return tmp.second;
    }

    // 2. 淘汰逻辑
    if (static_cast<int>(list.size()) >= capacity && !list.empty()) {
        destroyBlasHandle(list.front().second.handle);
        list.pop_front();
    }

    // 3. 统一创建 Handle，失败返回空 handle（本次调用失败，下次调用可重建）
    BlasCacheValue value;
    if (asdBlasCreate(value.handle) != AsdSip::ErrorType::ACL_SUCCESS) {
        return BlasCacheValue{};
    }

    // 4. 调用回调做 MakePlan；失败时销毁句柄且不入缓存，避免坏 handle 固化为持续故障（issue #128）
    if (makePlanFunc(value.handle) != AsdSip::ErrorType::ACL_SUCCESS) {
        asdBlasDestroy(value.handle);
        return BlasCacheValue{};
    }

    list.push_back(std::make_pair(cacheKey, value));
    return value;
}

void BlasMixCache::setCapacity(int64_t maxSize)
{
    std::lock_guard<std::mutex> guard(blasMutex);
    if (capacity == maxSize) {
        return;
    }

    capacity = maxSize;
    while (static_cast<int>(list.size()) > capacity && !list.empty()) {
        destroyBlasHandle(list.front().second.handle);
        list.pop_front();
    }
}

int64_t BlasMixCache::getCapacity() { return capacity; }

int64_t BlasMixCache::getSize() { return list.size(); }

void BlasMixCache::clear()
{
    std::lock_guard<std::mutex> guard(blasMutex);
    for (auto it = list.begin(); it != list.end(); ++it) {
        BlasPair tmp = *it;
        destroyBlasHandle(tmp.second.handle);
    }
    list.clear();
}
asdBlasHandle getBlasHandle(const std::string& opName, const std::vector<int64_t>& params,
                            const std::function<AspbStatus(asdBlasHandle)>& makePlanFunc)
{
    BlasCacheKey cacheKey;
    cacheKey.params = params;
    cacheKey.opName = opName;
    return blasCache.get(cacheKey, makePlanFunc).handle;
}
} // namespace op_api
