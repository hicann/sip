/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>
#include "params/fft_c2c_regbase.h"
#include "tiling/fft_c2c_large_kernel.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;

class FftC2CRegBaseOperation : public OperationBase {
public:
    explicit FftC2CRegBaseOperation(const std::string& opName) noexcept : OperationBase(opName) {}

    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "RegBase launch is inconsistent", return nullptr);
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950) {
            return nullptr;
        }
        if (launchParam.GetParam().Type() == typeid(OpParam::FftC2CLargeStage)) {
            const auto& p = AnyCast<OpParam::FftC2CLargeStage>(launchParam.GetParam());
            if (p.stage != OpParam::FftLargeStage::RegA && p.stage != OpParam::FftLargeStage::RegB)
                return nullptr;
            return GetKernelByName(OpParam::FftLargeKernelName(p.stage));
        }
        return GetKernelByName("FftC2CRegBaseC64Kernel");
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        if (specificParam.Type() == typeid(OpParam::FftC2CLargeStage))
            return 3;
        return specificParam.Type() == typeid(OpParam::FftC2CRegBase) ? 2 : 0;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        if (launchParam.GetParam().Type() == typeid(OpParam::FftC2CLargeStage)) {
            return InferFftLargeStageShape(launchParam, outTensors);
        }
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CRegBase),
                     "RegBase parameter type is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2 && outTensors.size() == 1, "RegBase tensor count is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        const auto& param = AnyCast<OpParam::FftC2CRegBase>(launchParam.GetParam());
        ASDSIP_CHECK(OpParam::IsFftC2CRegBaseSupported(param.fftN) && param.batchSize > 0,
                     "RegBase shape is unsupported", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = {param.batchSize, param.fftN};
        return Status::OkStatus();
    }
};

REG_OPERATION(FftC2CRegBaseOperation);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CRegBaseLargeAKernel, RegA, RegA);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CRegBaseLargeBKernel, RegB, RegB);
} // namespace AsdSip
