/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_plan.h"
#include "fft_kernel.h"
#include "fft_coeffs.h"
#include "fft_tiling.h"
#include "fft_tiling_def.h"

#include "helper.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace Fft {

// ---------------------------------------------------------------------------
// Plan / Exec / Destroy
// ---------------------------------------------------------------------------
bool FftPlanCreate(FftPlan& plan, int64_t b1, int64_t fftN1, int64_t fftN2, int32_t direction, int32_t deviceId,
                   int32_t coreNum)
{
    plan.b1 = b1;
    plan.fftN1 = fftN1;
    plan.fftN2 = fftN2;
    plan.direction = direction;
    plan.deviceId = deviceId;

    if (!ComputeTiling(plan.tiling, b1, fftN1, fftN2, direction, coreNum)) {
        return false;
    }
    plan.coeffsSize = GetCoeffsSize();
    plan.tilingSize = sizeof(FftTilingData);

    // ---- Build host coefficients (one buffer: twiddles + W50, both dirs) ----
    std::vector<float> hostCoeffs(plan.coeffsSize / sizeof(float));
    BuildAllCoeffs(hostCoeffs.data());

    // ---- Device setup ----
    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtSetDevice failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dCoeffs), plan.coeffsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc coeffs failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMemcpy(plan.dCoeffs, plan.coeffsSize, hostCoeffs.data(), plan.coeffsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: copy coeffs failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs);
        plan.dCoeffs = nullptr;
        return false;
    }

    // ---- Device tiling (kernel reads FftTilingData from GM) ----
    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dTiling), plan.tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc tiling failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs);
        plan.dCoeffs = nullptr;
        return false;
    }

    FftTilingData devTiling;
    FillDeviceTiling(plan.tiling, devTiling);

    ret = aclrtMemcpy(plan.dTiling, plan.tilingSize, &devTiling, plan.tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: copy tiling failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs);
        aclrtFree(plan.dTiling);
        plan.dCoeffs = nullptr;
        plan.dTiling = nullptr;
        return false;
    }

    std::cout << "FftPlanCreate: shape=[" << b1 << ", " << fftN1 << ", " << fftN2 << "]"
              << " direction=" << direction << " cores=" << coreNum << " (matrices/core: " << plan.tiling.matsPerCore
              << (plan.tiling.remCores > 0 ? " or " + std::to_string(plan.tiling.matsPerCore + 1) + " for " +
                                                 std::to_string(plan.tiling.remCores) + " cores" :
                                             "")
              << ")"
              << " coeffsSize=" << plan.coeffsSize << " twOff=" << devTiling.twiddleOffset << "B"
              << " w50Off=" << devTiling.w50Offset << "B"
              << " (float: " << devTiling.twiddleOffset / 4 << "/" << devTiling.w50Offset / 4 << ")"
              << " tilingSize=" << plan.tilingSize << " (fused AIC-GEMM + AIV-SIMT kernel, no workspace)" << std::endl;

    return true;
}

void FftExec2D(FftPlan& plan, uint8_t* dInput, uint8_t* dOutput, aclrtStream stream, int32_t coreNum)
{
    uint32_t blockDim = static_cast<uint32_t>(coreNum);
    FftC2C2DFusedKernelLaunch(dInput, dOutput, plan.dCoeffs, plan.dTiling, blockDim, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}

void FftPlanDestroy(FftPlan& plan)
{
    if (plan.dCoeffs != nullptr) {
        aclrtFree(plan.dCoeffs);
        plan.dCoeffs = nullptr;
    }
    if (plan.dTiling != nullptr) {
        aclrtFree(plan.dTiling);
        plan.dTiling = nullptr;
    }
}

} // namespace Fft
