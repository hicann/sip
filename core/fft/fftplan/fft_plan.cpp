/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftplan/fft_plan.h"

namespace AsdSip {

FFTPlan::FFTPlan() {}

void FFTPlan::markInitialized()
{
    // 初始化失败（PLAN_FAILED）的 plan 不允许被尾部 markInitialized 无条件
    // 覆写回 INITIALIZED——否则 isInitialized() 守卫失效，空 operation 的
    // plan 会在后续 GetWorkspaceSize/exec 中空指针崩溃（issue #169）
    if (fftStatus != AsdSip::asdFftPlanStatus::PLAN_FAILED) {
        fftStatus = AsdSip::asdFftPlanStatus::PLAN_INITIALIZED;
    }
}

void FFTPlan::markFailed() { fftStatus = AsdSip::asdFftPlanStatus::PLAN_FAILED; }

bool FFTPlan::isInitialized() const { return fftStatus == AsdSip::asdFftPlanStatus::PLAN_INITIALIZED; }

bool FFTPlan::hasWorkspace() const { return workspaceAddr != nullptr; }

bool FFTPlan::isForward() const { return direction == AsdSip::asdFftDirection::ASCEND_FFT_FORWARD; }

int64_t FFTPlan::eleNum() const
{
    int64_t num = batchSize;
    for (auto size : fftSizes) {
        num *= size;
    }
    return num;
}

} // namespace AsdSip
