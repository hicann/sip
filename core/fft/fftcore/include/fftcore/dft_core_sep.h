/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DFTCORESEP__
#define __DFTCORESEP__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class DFTCoreSep : public FFTCoreBase {
public:
    DFTCoreSep(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
               bool forward)
        : FFTCoreBase(FFTCoreType::kDft, nDone, nDoing, nLeft, batch, fftType, forward)
    {}
    ~DFTCoreSep() override { DestroyInDevice(); }
    void Run(void* inputReal, void* inputImag, void* outputReal, void* outputImag, void* stream,
             workspace::Workspace& workspace) override;
    bool SupportsSeparatedExec() const override { return true; } // 分离执行核心（issue #161）
    size_t EstimateWorkspaceSize() override;

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;
    AsdSip::AspbStatus InitRotationMatrix();
    AsdSip::AspbStatus InitTactic();

    std::string opName{"DftSepOperation"};
    std::shared_ptr<AsdSip::FFTensor> rotationMatrix;
    std::shared_ptr<AsdSip::FFTensor> rotationMatrixReal;
    std::shared_ptr<AsdSip::FFTensor> rotationMatrixImag;
};

#endif
