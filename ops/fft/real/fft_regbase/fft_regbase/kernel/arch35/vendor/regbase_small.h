/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_STANDALONE_REGBASE_SMALL_H
#define FFT_STANDALONE_REGBASE_SMALL_H
#include "acl/acl.h"
#include "fft_c2c_plan.h"
extern "C" bool LaunchFftC2CRegBaseSmall(float* input, float* twiddles, float* output, uint8_t* tiling,
                                         uint32_t blockDim, aclrtStream stream, FftC2CRegBaseSmallVariant variant,
                                         int32_t n, int32_t tilePoints, bool inverse);
#endif
