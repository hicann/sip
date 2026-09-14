/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DDD_SEP_CORE__
#define __DDD_SEP_CORE__

#include "ops.h"
#include "fftcore/fft_core_3d_base.h"
#include "utils/aspb_status.h"

class DddCoreSep : public FftCore3DBase {
public:
    DddCoreSep(unsigned fftX, unsigned fftY, unsigned fftZ, unsigned batch, AsdSip::asdFftType fftType, bool forward)
        : FftCore3DBase(FFTCoreType::kDd, fftX, fftY, fftZ, batch, fftType, forward)
    {}
    ~DddCoreSep() override {}
    size_t EstimateWorkspaceSize() override;
    void Run(void* inputReal, void* inputImag, void* outputReal, void* outputImag, void* stream,
             workspace::Workspace& workspace) override;
    bool SupportsSeparatedExec() const override { return true; } // 分离执行核心（issue #161）
private:
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;
    std::string opName{"DddSepOperation"};

    AsdSip::AspbStatus InitTactic();
    AsdSip::AspbStatus InitRotationMatrix(FFTCoreType coreType, int64_t fftN,
                                          std::shared_ptr<AsdSip::FFTensor>& rotMatPtr);
    std::shared_ptr<AsdSip::FFTensor> rotationMatrixX;
    std::shared_ptr<AsdSip::FFTensor> rotationMatrixY;
    std::shared_ptr<AsdSip::FFTensor> rotationMatrixZ;
};

#endif
