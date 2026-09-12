/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Selected from the experiment; provenance: ../SOURCE_MANIFEST.json.
#ifndef FFT_REGBASE_LARGE_H
#define FFT_REGBASE_LARGE_H

#include "acl/acl.h"

#include <cstdint>

// First distributed RegBase power-of-two C2C candidate:
//
//   N = 16384 = 128 * 128
//   input index  = n0 * 128 + n1
//   output index = k0 + 128 * k1
//
// Phase A performs the 128-point FFTs over n0 and writes
// workspace[k0][n1] after multiplying by W_N^(n1*k0). Phase B performs the
// 128-point FFTs over n1 and writes natural-order output. The phase movement
// groups are independent because the GM workspace layout does not depend on
// either group size. Both transforms are unnormalized. The host launches the
// two phases in one ACL stream, so the kernel boundary supplies execution and
// GM visibility ordering; the kernels contain no inter-AIV barrier or polling
// protocol.
//
// input, workspace, and output are batch-major interleaved FP32 complex
// tensors. localTwiddles contains 128 complex values: 127 compact radix-2
// Stockham twiddles plus one aligned padding value. crossTwiddles contains
// 128*128 complex values in n1-major order: [n1][k0].
struct alignas(8) FftC2CRegBaseLargeTilingData {
    int64_t batch;
    int32_t n;
    int32_t groupSize;
};

static_assert(sizeof(FftC2CRegBaseLargeTilingData) == 16, "RegBase large host/device tiling ABI must remain 16 bytes");

#endif
