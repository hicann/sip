
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "complex_mat_dot_tiling_data.h"
#include "complex_mat_dot.h"
#include "complex_mat_dot_tiling.h"

static constexpr uint32_t COMPLEX_NUM = 2;

namespace AsdSip {
using namespace Mki;

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus ComplexMatDotTiling(const LaunchParam& launchParam, KernelInfo& kernelInfo)
{
    void* tilingData = kernelInfo.GetTilingHostAddr();

    ComplexMatDotTilingData* tilingDataPtr = (ComplexMatDotTilingData*)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNumMatDot = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNumMatDot == 0) {
        cubeCoreNumMatDot = 1;
    }
    cubeCoreNumMatDot = cubeCoreNumMatDot > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNumMatDot;

    OpParam::ComplexMatDot param = AnyCast<OpParam::ComplexMatDot>(launchParam.GetParam());

    // matrix m x n in complex64 format
    uint32_t m = param.m;
    uint32_t n = param.n;

    uint64_t* startOffset = nullptr;
    try {
        startOffset = new uint64_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "ComplexMatDotTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t* calNum = nullptr;
    try {
        calNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "ComplexMatDotTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calNum + i) = 0;
    }

    // m*n 在 uint32_t 域内相乘可能溢出回绕（如 m=n=65536 时结果为 0），
    // 先提升到 uint64_t 再计算，避免 tiling 参数错误（issue #112）
    uint64_t totalNum = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    uint32_t numEachCore = static_cast<uint32_t>(totalNum / vecCoreNum); // 40 num of complex
    uint32_t remainNum = static_cast<uint32_t>(totalNum - static_cast<uint64_t>(vecCoreNum) * numEachCore);

    if (numEachCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            *(calNum + i) = 1;
            *(startOffset + i) = i * COMPLEX_NUM; // each row has 2*n FP32 elements
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currNum = numEachCore + 1;
            } else {
                currNum = numEachCore;
            }
            *(calNum + i) = currNum;
            *(startOffset + i) = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }

    tilingDataPtr->m = m; // num of rows
    tilingDataPtr->n = n; // num of FP32 elements each row

    auto ret = memcpy_s(tilingDataPtr->startOffset, vecCoreNum * sizeof(uint64_t), startOffset,
                        vecCoreNum * sizeof(uint64_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "startOffset memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ret = memcpy_s(tilingDataPtr->calNum, vecCoreNum * sizeof(uint32_t), calNum, vecCoreNum * sizeof(uint32_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "calNum memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    delete[] startOffset;
    startOffset = nullptr;
    delete[] calNum;
    calNum = nullptr;
    // Aicore : VectorCore : CubeCore = 1 : 2 : 1
    kernelInfo.SetBlockDim(cubeCoreNumMatDot); // use all aicores
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
