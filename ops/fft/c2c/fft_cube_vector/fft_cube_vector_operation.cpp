/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling/fft_c2c_large_kernel.h"

namespace AsdSip {
using namespace Mki;
class FftC2CCubeVectorOperation : public OperationBase {
public:
    explicit FftC2CCubeVectorOperation(const std::string& name) noexcept : OperationBase(name) {}
    Kernel* GetBestKernel(const LaunchParam& launch) const override
    {
        if (!IsConsistent(launch) || PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 ||
            launch.GetParam().Type() != typeid(OpParam::FftC2CLargeStage))
            return nullptr;
        const auto& p = AnyCast<OpParam::FftC2CLargeStage>(launch.GetParam());
        if (!OpParam::IsFftLargeStageSupported(p) || p.stage <= OpParam::FftLargeStage::RegB)
            return nullptr;
        return GetKernelByName(OpParam::FftLargeKernelName(p.stage));
    }
    int64_t GetInputNum(const Any& param) const override
    {
        return param.Type() == typeid(OpParam::FftC2CLargeStage) ? 3 : 0;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& output) const override
    {
        return InferFftLargeStageShape(launch, output);
    }
};
REG_OPERATION(FftC2CCubeVectorOperation);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2COuterKernel, OuterRegister, OuterRadix8);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CCubeVectorAKernel, CachedA, ExtendedA);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CCubeVectorBKernel, ExtendedB, ExtendedB);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CPackedKernel, Packed, Packed);
ASDSIP_DEFINE_FFT_LARGE_KERNEL(FftC2CGaussKernel, Gauss, Gauss);
} // namespace AsdSip
