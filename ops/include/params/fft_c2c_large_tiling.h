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

// The first 16 bytes preserve the imported RegBase batch/N/group ABI.
struct alignas(8) FftC2CLargeTilingData {
    int64_t batch;
    int32_t n;
    int32_t group;
    int32_t q;
    int32_t columns;
    int32_t inverse;
    int32_t reserved;
};
static_assert(sizeof(FftC2CLargeTilingData) == 32);
