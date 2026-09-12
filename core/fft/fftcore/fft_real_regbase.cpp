/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fftcore/fft_real_regbase_core.h"
#include "params/fft_real_regbase.h"
#include "params/fft_c2c_large.h"
#include "fft_c2c_coefficients.h"
#include "ops.h"
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace AsdSip;
using namespace Mki;
namespace {
enum class Slot { Input, Output, Scratch, CoreWork };
struct RealStage {
    LaunchParam launch;
    RunInfo run;
    std::unique_ptr<Kernel> kernel;
    std::vector<uint8_t> hostTiling;
    Slot input, output;
    ~RealStage()
    {
        if (run.GetTilingDeviceAddr())
            MkiRtMemFreeDevice(run.GetTilingDeviceAddr());
    }
};
} // namespace

struct FftRealRegBaseCore::Impl {
    fft::real::Descriptor descriptor;
    std::vector<std::shared_ptr<FFTensor>> coefficients;
    std::vector<std::unique_ptr<RealStage>> stages;
    Impl(unsigned n, unsigned batch, asdFftType type, bool forward)
        : descriptor(fft::real::Select(n, batch,
                                       type == asdFftType::ASCEND_FFT_C2R ? fft::real::Kind::C2R : fft::real::Kind::R2C,
                                       !forward, PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR)))
    {}
};

FftRealRegBaseCore::FftRealRegBaseCore(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, asdFftType type,
                                       bool forward)
    : FFTCoreBase(type == asdFftType::ASCEND_FFT_C2R ? kFftC2RRegBase : kFftR2CRegBase, nDone, nDoing, nLeft, batch,
                  type, forward),
      impl_(std::make_unique<Impl>(nDoing, batch, type, forward))
{}
FftRealRegBaseCore::~FftRealRegBaseCore() = default;

size_t FftRealRegBaseCore::EstimateWorkspaceSize()
{
    const auto& d = impl_->descriptor;
    // The last region supports public aliased buffers through device copy-back.
    return getAlignedSize(d.scratchBytes) + getAlignedSize(d.IsFused() ? 0 : d.core.workspaceBytes) +
           getAlignedSize(d.outputBytes);
}

bool FftRealRegBaseCore::PreAllocateInDevice()
{
    using S = OpParam::FftRealStage;
    const auto& d = impl_->descriptor;
    if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 || problemDesc.nDone != 1 ||
        problemDesc.nLeft != 1)
        return false;
    auto cache = [&](unsigned role, const std::function<std::vector<float>()>& build) {
        CoeffKey key = {coreType, role, {d.n}, problemDesc.forward};
        auto tensor = FFTensorCache::getCoeff(key, [&build]() -> FFTensor* {
            const auto values = build();
            auto result = std::make_unique<FFTensor>();
            result->desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {static_cast<int64_t>(values.size())}, {}, 0};
            result->dataSize = values.size() * sizeof(float);
            auto* host = new float[values.size()];
            std::copy(values.begin(), values.end(), host);
            result->hostData = host;
            return result.release();
        });
        impl_->coefficients.push_back(tensor);
        return tensor;
    };
    auto dummy = cache(0, [] { return std::vector<float>(1, 0); });
    auto boundary = cache(1, [&] { return fft::real::BuildBoundaryTwiddles(d); });
    auto local = cache(2, [&] { return fft::c2c::BuildLocalTwiddles(d.core); });
    auto initialize = [&](std::unique_ptr<RealStage> stage, const char* name) {
        auto* operation = Ops::Instance().GetOperationByName(name);
        if (!operation || !operation->InferShape(stage->launch).Ok())
            return false;
        stage->kernel.reset(operation->GetBestKernel(stage->launch));
        if (!stage->kernel || !stage->kernel->CanSupport(stage->launch))
            return false;
        stage->kernel->SetLaunchWithTiling(false);
        const auto bytes = stage->kernel->GetTilingSize(stage->launch);
        if (!bytes)
            return false;
        stage->hostTiling.resize(bytes, 0);
        stage->kernel->SetTilingHostAddr(stage->hostTiling.data(), bytes);
        if (!stage->kernel->Init(stage->launch).Ok())
            return false;
        void* deviceTiling = nullptr;
        if (MkiRtMemMallocDevice(&deviceTiling, bytes, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS)
            return false;
        stage->run.SetTilingDeviceAddr(static_cast<uint8_t*>(deviceTiling));
        if (MkiRtMemCopy(deviceTiling, bytes, stage->hostTiling.data(), bytes, MKIRT_MEMCOPY_HOST_TO_DEVICE) !=
            MKIRT_SUCCESS)
            return false;
        impl_->stages.push_back(std::move(stage));
        return true;
    };
    auto realStage = [&](S kind, Slot input, Slot output) {
        OpParam::FftRealRegBase p{d.n, d.batch, d.inverse ? 1 : 0, d.kind == fft::real::Kind::C2R ? 1 : 0, kind};
        auto stage = std::make_unique<RealStage>();
        stage->input = input;
        stage->output = output;
        Tensor x, y;
        x.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {d.batch, OpParam::RealSignalFloats(p, false)}, {}, 0};
        y.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {d.batch, OpParam::RealSignalFloats(p, true)}, {}, 0};
        x.dataSize = size_t(d.batch) * x.desc.dims[1] * sizeof(float);
        y.dataSize = size_t(d.batch) * y.desc.dims[1] * sizeof(float);
        stage->launch.SetParam(p);
        stage->launch.AddInTensor(x);
        stage->launch.AddInTensor(*(kind == S::Boundary ? dummy : local));
        stage->launch.AddInTensor(*(kind == S::Small ? dummy : boundary));
        stage->launch.AddOutTensor(y);
        return initialize(std::move(stage), "FftRealRegBaseOperation");
    };
    auto core = [&](Slot input, Slot output) {
        if (!d.core.Large())
            return realStage(S::Small, input, output);
        auto cross = cache(3, [&] {
            constexpr double twoPi = 6.283185307179586476925286766559;
            std::vector<float> values(2 * d.core.n);
            for (int r = 0; r < d.core.q; ++r)
                for (int c = 0; c < d.core.p; ++c) {
                    const double angle = (d.inverse ? 1.0 : -1.0) * twoPi * r * c / d.core.n;
                    values[2 * (r * d.core.p + c)] = std::cos(angle);
                    values[2 * (r * d.core.p + c) + 1] = std::sin(angle);
                }
            return values;
        });
        for (auto kind : {OpParam::FftLargeStage::RegA, OpParam::FftLargeStage::RegB}) {
            const bool first = kind == OpParam::FftLargeStage::RegA;
            auto stage = std::make_unique<RealStage>();
            stage->input = first ? input : Slot::CoreWork;
            stage->output = first ? Slot::CoreWork : output;
            stage->launch.SetParam(OpParam::FftC2CLargeStage{d.core.n, d.batch, d.inverse ? 1 : 0, kind});
            Tensor signal;
            signal.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {d.batch, d.core.n}, {}, 0};
            signal.dataSize = d.scratchBytes;
            stage->launch.AddInTensor(signal);
            stage->launch.AddInTensor(*local);
            stage->launch.AddInTensor(*(first ? cross : dummy));
            stage->launch.AddOutTensor(signal);
            if (!initialize(std::move(stage), "FftC2CRegBaseOperation"))
                return false;
        }
        return true;
    };
    if (d.IsFused()) {
        if (!realStage(S::Fused, Slot::Input, Slot::Output))
            return false;
    } else if (d.kind == fft::real::Kind::R2C) {
        if (!core(Slot::Input, Slot::Scratch) || !realStage(S::Boundary, Slot::Scratch, Slot::Output))
            return false;
    } else {
        if (!realStage(S::Boundary, Slot::Input, Slot::Scratch) || !core(Slot::Scratch, Slot::Output))
            return false;
    }
    const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
    if (trace && std::string(trace) == "1") {
        std::cerr << "A5_REAL_FFT_VARIANT kind=" << fft::real::KindName(d.kind)
                  << " variant=" << fft::real::VariantName(d.variant) << " N=" << d.n << " effective_batch=" << d.batch
                  << " inverse=" << d.inverse << " launches=" << impl_->stages.size()
                  << " workspace_bytes=" << EstimateWorkspaceSize() << " fft_workspace_bytes=" << d.workspaceBytes
                  << " fused_ub_bytes=" << d.fusedUbBytes << " slots=" << d.fusedUbSlots
                  << " boundary_tile=" << d.boundaryTilePoints << '\n';
        for (const auto& stage : impl_->stages) {
            std::cerr << "A5_REAL_FFT_STAGE stage=" << stage->kernel->GetName()
                      << " key=" << stage->kernel->GetKernelInfo().GetTilingId()
                      << " block_dim=" << stage->kernel->GetKernelInfo().GetBlockDim() << '\n';
        }
    }
    return true;
}

void FftRealRegBaseCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    Run(input.data, output.data, stream, workspace);
}
void FftRealRegBaseCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    const auto& d = impl_->descriptor;
    const auto in = reinterpret_cast<uintptr_t>(input), out = reinterpret_cast<uintptr_t>(output);
    if (!in || !out || in % 32 || out % 32 ||
        (in != out && (in < out ? out - in < d.inputBytes : in - out < d.outputBytes)))
        throw std::invalid_argument("Real FFT requires aligned disjoint buffers or exact aliasing");
    auto* arena = static_cast<uint8_t*>(workspace.allocate(EstimateWorkspaceSize()));
    const size_t workOffset = getAlignedSize(d.scratchBytes);
    const size_t aliasOffset = workOffset + getAlignedSize(d.IsFused() ? 0 : d.core.workspaceBytes);
    const bool alias = input == output;
    auto address = [&](Slot slot) -> void* {
        switch (slot) {
            case Slot::Input:
                return input;
            case Slot::Output:
                return alias ? arena + aliasOffset : output;
            case Slot::Scratch:
                return arena;
            case Slot::CoreWork:
                return arena + workOffset;
        }
        return nullptr;
    };
    try {
        for (auto& stage : impl_->stages) {
            stage->launch.GetInTensor(0).data = address(stage->input);
            stage->launch.GetOutTensor(0).data = address(stage->output);
            stage->run.SetStream(stream);
            stage->run.SetScratchDeviceAddr(static_cast<uint8_t*>(address(stage->output)));
            if (!stage->kernel->Run(stage->launch, stage->run).Ok())
                throw std::runtime_error("Real RegBase stage launch failed");
        }
        if (alias && MkiRtMemCopyAsync(output, d.outputBytes, arena + aliasOffset, d.outputBytes,
                                       MKIRT_MEMCOPY_DEVICE_TO_DEVICE, stream) != MKIRT_SUCCESS)
            throw std::runtime_error("Real RegBase alias copy-back failed");
    } catch (...) {
        workspace.recycleLast();
        throw;
    }
    workspace.recycleLast();
}
