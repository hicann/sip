/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLAS_COMMON_OP_API_H
#define BLAS_COMMON_OP_API_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "blas_api.h"

using namespace AsdSip;

namespace op_api {

// 由于 BLAS 参数各异，使用通用的 Key 代替固定的 Param
struct BlasCacheKey {
    std::string opName;
    std::vector<int64_t> params;
};

inline bool operator==(const BlasCacheKey& one, const BlasCacheKey& other)
{
    return one.opName == other.opName && one.params == other.params;
}

struct BlasCacheValue {
    asdBlasHandle handle = nullptr;
};

class BlasMixCache {
public:
    using BlasPair = std::pair<BlasCacheKey, BlasCacheValue>;
    explicit BlasMixCache(int64_t c) noexcept;

    // 因为 BLAS 的 create 逻辑各异，需要传入 onMiss 回调函数；回调返回 plan 创建结果（issue #128）
    BlasCacheValue get(BlasCacheKey& cacheKey, std::function<AspbStatus(asdBlasHandle)> makePlanFunc);

    void setCapacity(int64_t maxSize);
    int64_t getCapacity();
    int64_t getSize();
    void clear();

private:
    int64_t capacity;
    std::list<BlasPair> list{};
    std::mutex blasMutex;
};

// 统一的销毁逻辑 (同步 + 销毁)
inline void destroyBlasHandle(asdBlasHandle handle)
{
    if (handle != nullptr) {
        asdBlasSynchronize(handle);
        asdBlasDestroy(handle);
    }
}
asdBlasHandle getBlasHandle(const std::string& opName, const std::vector<int64_t>& params,
                            const std::function<AspbStatus(asdBlasHandle)>& makePlanFunc);

} // namespace op_api

#endif // BLAS_COMMON_OP_API_H
