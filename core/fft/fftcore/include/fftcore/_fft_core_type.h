/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_BASE_CORE_TYPE__
#define __FFTCORE_BASE_CORE_TYPE__

enum FFTCoreType : unsigned {
    kAny,
    kFftB,
    kFftN,
    kFftStride,
    kFftMix,
    kDft,
    kDftC2R,
    kDftR2C,
    kFftC2R,
    kFftR2C,
    kFftBC2R,
    kFftBR2C,
    kDd,
    kDftSep,
    kDdd,
    kDddSep,
    kFftBSep,
    kFftC2RArch35,
    kFftR2CArch35,
    kFftC2CArch35,
    kFftC2C2DArch35,
    kFftC2CRegBase,
    kFftC2CCubeVector,
    kFftR2CRegBase,
    kFftC2RRegBase,
    kFftR2CStockham,
    kFftC2RStockham,

    // Preserve the original numeric value while naming the A5 backend explicitly.
    kFftC2CSimt = kFftC2CArch35,

};

#endif
