/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_B_SEP__
#define __FFTCORE_B_SEP__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

class FFTCoreBSep : public FFTCoreBase {
public:
    FFTCoreBSep(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
                bool forward)
        : FFTCoreBase(FFTCoreType::kFftBSep, nDone, nDoing, nLeft, batch, fftType, forward)
    {}
    ~FFTCoreBSep() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    // void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) override;
    void Run(void* inputReal, void* inputImag, void* outputReal, void* outputImag, void* stream,
             workspace::Workspace& workspace) override;
    bool SupportsSeparatedExec() const override { return true; } // 分离执行核心（issue #161）

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;

    Mki::Any InitParam();
    AsdSip::AspbStatus InitWMatrix(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward);
    AsdSip::AspbStatus InitTMatrix(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward);
    AsdSip::AspbStatus InitTactic();

    std::shared_ptr<AsdSip::FFTensor> wMatrix;
    std::shared_ptr<AsdSip::FFTensor> tMatrix;
};

#endif
