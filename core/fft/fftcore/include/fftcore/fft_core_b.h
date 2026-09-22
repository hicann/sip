/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_B__
#define __FFTCORE_B__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class FFTCoreB : public FFTCoreBase {
public:
    FFTCoreB(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
             bool forward)
        : FFTCoreBase(FFTCoreType::kFftB, nDone, nDoing, nLeft, batch, fftType, forward)
    {
    }
    ~FFTCoreB() override
    {
        DestroyInDevice();
    }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) override;
    void Run(void *input, void *output, void *stream, workspace::Workspace &workspace) override;
    bool OpsFftBackendAdapted() const override { return true; }
    bool OpsFftBackendStubReady() const override;

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice();
    // ops-fft kernel 直调后端 (SIP_FFT_BACKEND=ops-fft)
    void RunViaOpsFft(void *input, void *output, void *stream, workspace::Workspace &workspace);
    AsdSip::AspbStatus InitIndex();
    AsdSip::AspbStatus InitWMatrix();
    AsdSip::AspbStatus InitTMatrix();
    AsdSip::AspbStatus InitTactic();
    std::shared_ptr<AsdSip::FFTensor> wMatrix;
    std::shared_ptr<AsdSip::FFTensor> tMatrix;
    std::shared_ptr<AsdSip::FFTensor> index;
    // ops-fft 后端: 常量 device 缓存 (首次 Run 时上传, plan 级生命周期)
    void *opsWMatrix = nullptr;
    void *opsTMatrix = nullptr;
    void *opsIndex = nullptr;
    uint32_t opsCachedBlocks = 0;
    uint8_t *opsCachedSync = nullptr;
};

#endif