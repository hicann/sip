/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_ISTFT_SPLICE__
#define __FFTCORE_ISTFT_SPLICE__

#include <vector>
#include "ops.h"
#include "utils/aspb_status.h"
#include "fftplan/fft_plan.h"
#include "fftoperation/fft_operation.h"
#include "fft_api.h"

enum class FftNormMode {
    NONE,      // No normalization
    BY_ROOT_N, // Divide by sqrt(signal_size)
    BY_N,      // Divide by signal_size
};

class FFTCoreIstftAny : public FftOperation {
public:
    explicit FFTCoreIstftAny(const struct AsdSip::IstftDesc istftParamDesc) : istftAnyDesc{istftParamDesc} {}
    bool init() override
    {
        ASDSIP_LOG(DEBUG) << "Success to initialize istft fftcore.";
        return true;
    };
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override {};
    void Run(Tensor& input, Tensor& window, Tensor& output, void* stream, workspace::Workspace& workspace) override;
    void Run(void* input, void* window, void* output, void* stream, workspace::Workspace& workspace) override;
    size_t EstimateWorkspaceSize() override;
    ~FFTCoreIstftAny() override {}

protected:
    struct AsdSip::IstftDesc istftAnyDesc;

private:
    uint8_t* tempCP64Buffer = nullptr;     // 缓存temp tensor
    uint8_t* deviceBuffer = nullptr;       // 小算子的workspace
    uint8_t* unfoldGradBuffer = nullptr;   // unFoldGrad算子的workspace
    uint8_t* ySliceBuffer = nullptr;       // unFoldGrad算子的workspace
    uint8_t* windowSliceeBuffer = nullptr; // 窗口包络 w^2 的 scratch（避免回写用户 window 存储）
    uint8_t* windowExpandBuffer = nullptr; // unFoldGrad算子的workspace
    uint8_t* tempYBuffer = nullptr;

    size_t ComputerUnfoldBufferSize();
    size_t WindowExpandSize();
    size_t SliceSize();
    size_t TempCp64Size();
    size_t TempYSize();
};

#endif
