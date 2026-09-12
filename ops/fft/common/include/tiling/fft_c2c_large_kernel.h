/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/module/module.h>
#include <mki_loader/op_register.h>
#include "params/fft_c2c_large.h"
#include "params/fft_c2c_large_tiling.h"
#include "utils/assert.h"

namespace AsdSip {
inline Mki::Status InferFftLargeStageShape(const Mki::LaunchParam& launch, Mki::SVector<Mki::Tensor>& outputs)
{
    ASDSIP_CHECK(launch.GetParam().Type() == typeid(OpParam::FftC2CLargeStage) && launch.GetInTensorCount() == 3 &&
                     outputs.size() == 1,
                 "Large C2C stage expects three inputs and one output",
                 return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR));
    const auto& p = Mki::AnyCast<OpParam::FftC2CLargeStage>(launch.GetParam());
    ASDSIP_CHECK(OpParam::IsFftLargeStageSupported(p), "Unsupported large C2C stage",
                 return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR));
    outputs[0].desc = launch.GetInTensor(0).desc;
    outputs[0].desc.dtype = Mki::TENSOR_DTYPE_COMPLEX64;
    outputs[0].desc.dims = {p.batchSize, p.fftN};
    return Mki::Status::OkStatus();
}

class FftC2CLargeStageKernel : public Mki::KernelBase {
public:
    FftC2CLargeStageKernel(const std::string& name, const Mki::BinHandle* handle, OpParam::FftLargeStage first,
                           OpParam::FftLargeStage last)
        : KernelBase(name, handle), first_(first), last_(last)
    {}

    bool CanSupport(const Mki::LaunchParam& launch) const override
    {
        using namespace Mki;
        ASDSIP_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_950 &&
                         launch.GetParam().Type() == typeid(OpParam::FftC2CLargeStage),
                     "Large C2C stages require Ascend 950 and matching parameters", return false);
        const auto& p = AnyCast<OpParam::FftC2CLargeStage>(launch.GetParam());
        ASDSIP_CHECK(OpParam::IsFftLargeStageSupported(p) && p.stage >= first_ && p.stage <= last_,
                     "Invalid large C2C stage/shape", return false);
        ASDSIP_CHECK(launch.GetInTensorCount() == 3 && launch.GetOutTensorCount() == 1,
                     "Invalid large C2C tensor count", return false);
        const auto& x = launch.GetInTensor(0);
        const auto& y = launch.GetOutTensor(0);
        ASDSIP_CHECK(x.desc.dtype == TENSOR_DTYPE_COMPLEX64 && y.desc.dtype == TENSOR_DTYPE_COMPLEX64 &&
                         x.desc.dims.size() == 2 && x.desc.dims[0] == p.batchSize && x.desc.dims[1] == p.fftN &&
                         y.desc.dims == x.desc.dims && x.IsContiguous() && y.IsContiguous() && x.desc.offset == 0 &&
                         y.desc.offset == 0,
                     "Large C2C signals require contiguous complex64 [batch,N]", return false);
        ASDSIP_CHECK(x.data == nullptr || y.data == nullptr || x.data != y.data,
                     "Each large C2C stage requires separate input and output", return false);
        for (int i = 1; i <= 2; ++i) {
            const auto& coeff = launch.GetInTensor(i);
            ASDSIP_CHECK(coeff.desc.dtype == TENSOR_DTYPE_FLOAT && coeff.desc.dims.size() == 1 &&
                             coeff.desc.dims[0] == OpParam::FftLargeCoefficientCount(p, i) && coeff.IsContiguous() &&
                             coeff.desc.offset == 0,
                         "Large C2C coefficient layout mismatch", return false);
        }
        return true;
    }

    uint64_t GetTilingSize(const Mki::LaunchParam&) const override { return sizeof(FftC2CLargeTilingData); }

    Mki::Status RunWithArgs(void* args, void* stream, bool isDeviceAddr) override
    {
        using namespace Mki;
        if (kernelInfo_.GetLocalMemorySize() == 0) {
            return KernelBase::RunWithArgs(args, stream, isDeviceAddr);
        }
        // The bundled MKI KernelBase passes nullptr as the runtime config even
        // after SetLocalMemorySize. These ASC stages address dynamic UB, so the
        // launch must carry the same byte count as the standalone <<<...>>>.
        // Keep MKI's argument builder and registration, overriding only launch.
        constexpr uint32_t argumentCount = 7; // three inputs, output, workspace, tiling, overflow
        constexpr uint32_t tilingArgument = 5;
        const auto* binary = GetBinHandle();
        ASDSIP_CHECK(binary != nullptr && args != nullptr && kernelInfo_.GetConstTensorCount() == 0 &&
                         kernelInfo_.GetArgsSize() >= argumentCount * sizeof(void*) &&
                         kernelInfo_.GetArgsSize() <= UINT32_MAX,
                     "Invalid large FFT runtime arguments", return Status::FailStatus(ERROR_INVALID_VALUE));
        RtArgsExT argsEx{};
        argsEx.args = args;
        argsEx.argsSize = static_cast<uint32_t>(kernelInfo_.GetArgsSize());
        argsEx.isNoNeedH2DCopy = isDeviceAddr ? 1 : 0;
        if (kernelInfo_.GetLaunchWithTiling()) {
            argsEx.hasTiling = 1;
            argsEx.tilingAddrOffset = tilingArgument * sizeof(void*);
            argsEx.tilingDataOffset = argumentCount * sizeof(void*);
        }
        MkiRtKernelParam launch{};
        launch.tilingId = kernelInfo_.GetTilingId();
        launch.blockDim = kernelInfo_.GetBlockDim();
        launch.argsEx = &argsEx;
        RtTaskCfgInfoT config{};
        config.localMemorySize = kernelInfo_.GetLocalMemorySize();
        config.schemMode = static_cast<uint8_t>(kernelInfo_.GetScheduleMode());
        const auto handle = binary->GetHandle();
        const int result = *handle != nullptr ? MkiRtFunctionLaunchWithHandle(*handle, &launch, stream, &config) :
                                                MkiRtFunctionLaunchWithFlag(handle, &launch, stream, &config);
        ASDSIP_CHECK(result == MKIRT_SUCCESS, "Large FFT launch with local memory failed: " + std::to_string(result),
                     return Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR));
        return Status::OkStatus();
    }

    Mki::Status InitImpl(const Mki::LaunchParam& launch) override
    {
        using namespace Mki;
        using S = OpParam::FftLargeStage;
        ASDSIP_CHECK(CanSupport(launch), "Invalid large C2C launch", return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& p = AnyCast<OpParam::FftC2CLargeStage>(launch.GetParam());
        auto& platform = PlatformInfo::Instance();
        const int64_t aiv = platform.GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        const int64_t aic = platform.GetCoreNum(CoreType::CORE_TYPE_CUBE);
        auto* t = reinterpret_cast<FftC2CLargeTilingData*>(kernelInfo_.GetTilingHostAddr());
        ASDSIP_CHECK(t != nullptr && aiv > 0, "Missing tiling or vector cores",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        *t = {p.batchSize, static_cast<int32_t>(p.fftN), 0, static_cast<int32_t>(p.fftN / 16384), 0, p.isInverse, 0};
        int64_t blocks = aiv;
        uint64_t key = static_cast<uint64_t>(p.stage);
        uint32_t localMemory = 0;
        if (p.stage == S::RegA || p.stage == S::RegB) {
            const int64_t factorP = p.fftN <= 32768 ? 128 : 256;
            const int64_t factorQ = p.fftN / factorP;
            t->group = p.stage == S::RegA ? (p.fftN == 131072 && p.batchSize == 1 ? 16 : 4) :
                                            (p.fftN == 16384 ? 16 : 4);
            blocks = std::min(aiv, p.batchSize * (p.stage == S::RegA ? factorQ : factorP) / t->group);
            key = p.fftN + (p.stage == S::RegA && p.fftN == 131072 && t->group == 16 ? 1 : 0);
        } else if (p.stage >= S::OuterRegister && p.stage <= S::OuterRadix8) {
            t->columns = 4096 / t->q;
            while (t->columns > 64 && t->columns > std::max<int64_t>(64, p.batchSize * 16384 / aiv)) {
                t->columns /= 2;
            }
            localMemory = p.stage <= S::OuterSimd ? 98816 : 164352;
        } else if (p.stage == S::CachedA || p.stage == S::ExtendedA) {
            blocks = std::min(aiv, p.batchSize * t->q * 32);
        } else if (p.stage == S::ExtendedB) {
            blocks = std::min(aiv, p.batchSize * t->q * 8);
            key = 0;
        } else {
            ASDSIP_CHECK(aic > 0 && aiv >= 2 * aic, "Mixed FFT requires paired AIC/AIV resources",
                         return Status::FailStatus(ERROR_INVALID_VALUE));
            blocks = std::min(aic, p.batchSize * t->q * 4);
            key = 0;
            localMemory = 132096;
        }
        kernelInfo_.SetBlockDim(static_cast<uint32_t>(blocks));
        kernelInfo_.SetTilingId(key);
        kernelInfo_.SetLocalMemorySize(localMemory);
        // MKI retains one pointer for this placeholder before the tiling pointer.
        kernelInfo_.GetScratchSizes() = {0};
        return Status::OkStatus();
    }

private:
    OpParam::FftLargeStage first_, last_;
};

#define ASDSIP_DEFINE_FFT_LARGE_KERNEL(Name, First, Last)                                                       \
    class Name : public FftC2CLargeStageKernel {                                                                \
    public:                                                                                                     \
        Name(const std::string& name, const Mki::BinHandle* handle) noexcept                                    \
            : FftC2CLargeStageKernel(name, handle, OpParam::FftLargeStage::First, OpParam::FftLargeStage::Last) \
        {}                                                                                                      \
    };                                                                                                          \
    REG_KERNEL_BASE(Name)
} // namespace AsdSip
