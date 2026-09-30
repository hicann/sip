/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <unordered_map>
#include "fftoperation/fft_operation.h"

#include "fft_api.h"
#include "utils/workspace.h"
namespace AsdSip {

struct IstftDesc {
    // istft in tensor ands out tensor dims
    int64_t channel;
    int64_t nFrames;
    int64_t fftSize;
    int64_t outSignalLen; // 输出参数
    int64_t windowDtype;  // 输出参数 ACL_COMPLEX64:16  ACL_FLOAT:0

    // sip库1d fft入参参数
    int64_t sipFftSize;
    int64_t sipFftBatch;

    // istft other params
    int64_t nFft;
    int64_t hopLengthOpt;
    int64_t winLengthOpt;
    bool center = true;
    bool normalized = false;
    bool onesidedOpt = false;
    int64_t lengthOpt = 0;
    bool returnComplex = true;
};

enum class asdFftPlanStatus { PLAN_UNINITIALIZED = 0, PLAN_INITIALIZED = 1, PLAN_FAILED = 2 };

struct PlanStep {
    std::unique_ptr<FftOperation> operation;
};

class FFTPlan {
public:
    asdFftPlanStatus fftStatus = AsdSip::asdFftPlanStatus::PLAN_UNINITIALIZED;
    asdFftType fftType = ASCEND_FFT_C2C;
    asdFftDirection direction = ASCEND_FFT_FORWARD;
    void* stream = nullptr;
    int64_t batchSize = 0;
    std::vector<int64_t> fftSizes;
    std::vector<int64_t> fftStrides;
    std::vector<PlanStep> steps;
    void* workspaceAddr = nullptr;
    size_t workspaceSize = 0; // make 后由 GetWorkspaceSize 记录的需求量，用于 SetWorkspace 后的越界防护（issue #195）

    FFTPlan();
    void markInitialized();
    void markFailed();
    bool isInitialized() const;
    bool hasWorkspace() const;
    bool isForward() const;
    int64_t eleNum() const;

    struct IstftDesc istftDesc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // istft 相关参数存储
};

} // namespace AsdSip
