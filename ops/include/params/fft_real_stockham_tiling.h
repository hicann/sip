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

namespace AsdSip {
struct FftRealStockhamTiling {
    int64_t batchSize;
    int64_t fftN;
    uint64_t workspaceOffsets[2];
    int32_t radixListLen;
    int32_t isOddN;
};
static_assert(sizeof(FftRealStockhamTiling) == 40, "Real Stockham tiling ABI");
} // namespace AsdSip
