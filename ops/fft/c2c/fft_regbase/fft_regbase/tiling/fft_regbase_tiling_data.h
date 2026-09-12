/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_FFT_REGBASE_TILING_DATA_H
#define ASDSIP_FFT_REGBASE_TILING_DATA_H
#include <cstdint>
namespace AsdSip {
struct alignas(8) FftC2CRegBaseTilingData {
    int64_t batch;
    int32_t n;
    int32_t tilePoints;
};
static_assert(sizeof(FftC2CRegBaseTilingData) == 16, "RegBase tiling ABI must remain 16 bytes");
} // namespace AsdSip
#endif
