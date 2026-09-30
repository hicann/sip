/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DD_CORE__
#define __DD_CORE__

#include "ops.h"
#include "fftcore/fft_core_2d_base.h"
#include "utils/aspb_status.h"

class DdCore : public FftCore2DBase {
public:
    DdCore(unsigned fftX, unsigned fftY, unsigned batch, AsdSip::asdFftType fftType, bool forward)
        : FftCore2DBase(FFTCoreType::kDd, fftX, fftY, batch, fftType, forward)
    {}
    ~DdCore() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace) override;

private:
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;

    std::string opName{"DdOperation"};
    AsdSip::AspbStatus InitPMatrix();
    AsdSip::AspbStatus InitQMatrix();
    AsdSip::AspbStatus InitTactic();
    std::shared_ptr<AsdSip::FFTensor> pMatrix;
    std::shared_ptr<AsdSip::FFTensor> qMatrix;
};

#endif
