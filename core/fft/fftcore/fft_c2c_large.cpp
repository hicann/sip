/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fftcore/fft_c2c_large_core.h"
#include "params/fft_c2c_large.h"
#include "params/fft_c2c_large_tiling.h"
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
using Route = fft::a5::Route;
using Kind = OpParam::FftLargeStage;
enum class BufferSlot { Input, Output, Work, Temporary };
constexpr double kTwoPi = 6.283185307179586476925286766559;

struct Stage {
    std::vector<uint8_t> hostTiling;
    LaunchParam launch;
    RunInfo run;
    std::unique_ptr<Kernel> kernel;
    BufferSlot input, output;
    ~Stage()
    {
        if (run.GetTilingDeviceAddr())
            MkiRtMemFreeDevice(run.GetTilingDeviceAddr());
    }
};

std::vector<float> LocalTwiddles(int length, bool inverse, size_t count)
{
    std::vector<float> values(count, 0);
    const double sign = inverse ? 1.0 : -1.0;
    for (int size = 2; size <= length; size *= 2) {
        for (int j = 0; j < size / 2; ++j) {
            const double angle = sign * kTwoPi * j / size;
            const size_t i = 2 * (size / 2 - 1 + j);
            values[i] = std::cos(angle);
            values[i + 1] = std::sin(angle);
        }
    }
    return values;
}

std::vector<float> CrossTwiddles(int rows, int columns, bool inverse)
{
    std::vector<float> values(2 * rows * columns);
    const double sign = inverse ? 1.0 : -1.0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < columns; ++c) {
            const double angle = sign * kTwoPi * r * c / (rows * columns);
            const size_t i = 2 * (r * columns + c);
            values[i] = std::cos(angle);
            values[i + 1] = std::sin(angle);
        }
    return values;
}

std::vector<float> CubeCoefficients(bool gauss, bool inverse)
{
    std::vector<float> values(gauss ? 3328 : 4352, 0);
    const double sign = inverse ? 1.0 : -1.0;
    for (int j = 0; j < 32; ++j)
        for (int k = 0; k < 32; ++k) {
            const double angle = sign * kTwoPi * j * k / 32;
            const float c = std::cos(angle), s = std::sin(angle);
            if (gauss) {
                values[j * 32 + k] = c;
                values[1024 + j * 32 + k] = s;
                values[2048 + j * 32 + k] = c + s;
            } else {
                values[2 * j * 64 + 2 * k] = c;
                values[2 * j * 64 + 2 * k + 1] = s;
                values[(2 * j + 1) * 64 + 2 * k] = -s;
                values[(2 * j + 1) * 64 + 2 * k + 1] = c;
            }
        }
    for (int s = 0; s < 4; ++s)
        for (int k = 0; k < 32; ++k) {
            const double angle = sign * kTwoPi * s * k / 128;
            const size_t i = (gauss ? 3072 : 4096) + 2 * (s * 32 + k);
            values[i] = std::cos(angle);
            values[i + 1] = std::sin(angle);
        }
    return values;
}
} // namespace

struct FftC2CLargeCore::Impl {
    Route route;
    size_t signalBytes;
    bool separateOuter;
    std::vector<std::shared_ptr<FFTensor>> coefficients;
    std::vector<std::unique_ptr<Stage>> stages;

    Impl(unsigned n, unsigned batch)
        : route(fft::a5::Select(n, batch)),
          signalBytes(static_cast<size_t>(n) * batch * 8),
          separateOuter(route != Route::Regbase && n > 16384)
    {}
};

FftC2CLargeCore::FftC2CLargeCore(FFTCoreType type, unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch,
                                 asdFftType fftType, bool forward)
    : FFTCoreBase(type, nDone, nDoing, nLeft, batch, fftType, forward), impl_(std::make_unique<Impl>(nDoing, batch))
{}

FftC2CLargeCore::~FftC2CLargeCore() = default;

size_t FftC2CLargeCore::EstimateWorkspaceSize()
{
    // Each stage reads its preceding buffer. Even aliased public input/output
    // need no copy-back: the last stage consumes only planner workspace.
    return getAlignedSize(impl_->signalBytes) * (impl_->separateOuter ? 2 : 1);
}

bool FftC2CLargeCore::PreAllocateInDevice()
{
    const int n = problemDesc.nDoing;
    const int64_t batch = problemDesc.batch;
    const bool inverse = !problemDesc.forward;
    const Route route = impl_->route;
    if (n < 16384 || route == Route::Existing || route == Route::Dft)
        return false;
    if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950)
        return false;
    const auto aiv = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);

    auto cache = [&](unsigned role, const std::function<std::vector<float>()>& build) {
        CoeffKey key = {coreType, role, {n, static_cast<int64_t>(route)}, problemDesc.forward};
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
    auto addStage = [&](Kind kind, BufferSlot inputSlot, BufferSlot outputSlot, const std::shared_ptr<FFTensor>& tw,
                        const std::shared_ptr<FFTensor>& cross) {
        auto stage = std::make_unique<Stage>();
        stage->input = inputSlot;
        stage->output = outputSlot;
        OpParam::FftC2CLargeStage param{n, batch, inverse ? 1 : 0, kind};
        if (!OpParam::IsFftLargeStageSupported(param))
            return false;
        Tensor signal;
        signal.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, n}, {}, 0};
        signal.dataSize = impl_->signalBytes;
        stage->launch.SetParam(param);
        stage->launch.AddInTensor(signal);
        stage->launch.AddInTensor(*tw);
        stage->launch.AddInTensor(*cross);
        stage->launch.AddOutTensor(signal);
        const char* operationName = kind <= Kind::RegB ? "FftC2CRegBaseOperation" : "FftC2CCubeVectorOperation";
        auto* operation = Ops::Instance().GetOperationByName(operationName);
        if (!operation || !operation->InferShape(stage->launch).Ok())
            return false;
        stage->kernel.reset(operation->GetBestKernel(stage->launch));
        if (!stage->kernel || !stage->kernel->CanSupport(stage->launch))
            return false;
        stage->kernel->SetLaunchWithTiling(false);
        const auto bytes = stage->kernel->GetTilingSize(stage->launch);
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

    if (route == Route::Regbase) {
        const auto descriptor = fft::c2c::Select(n, batch, inverse, aiv);
        auto tw = cache(1, [&] { return fft::c2c::BuildLocalTwiddles(descriptor); });
        auto cross = cache(2, [&] { return CrossTwiddles(descriptor.q, descriptor.p, inverse); });
        if (!addStage(Kind::RegA, BufferSlot::Input, BufferSlot::Work, tw, cross) ||
            !addStage(Kind::RegB, BufferSlot::Work, BufferSlot::Output, tw, dummy))
            return false;
    } else {
        const int q = n / 16384;
        const bool vector = route == Route::VectorExtended;
        const bool gauss = route == Route::GaussSimd || route == Route::GaussRadix4 || route == Route::GaussRadix8;
        auto tw = cache(1, [&] { return LocalTwiddles(128, inverse, 256); });
        auto cross = cache(2, [&] { return CrossTwiddles(128, 128, inverse); });
        BufferSlot innerInput = BufferSlot::Input;
        if (impl_->separateOuter) {
            auto outerTw = cache(3, [&] { return LocalTwiddles(q, inverse, 128); });
            auto outerCross = cache(4, [&] { return CrossTwiddles(q, 16384, inverse); });
            const Kind outer = vector                      ? Kind::OuterRegister :
                               route == Route::GaussRadix4 ? Kind::OuterRadix4 :
                               route == Route::GaussRadix8 ? Kind::OuterRadix8 :
                                                             Kind::OuterSimd;
            if (!addStage(outer, BufferSlot::Input, BufferSlot::Temporary, outerTw, outerCross))
                return false;
            innerInput = BufferSlot::Temporary;
        }
        if (!addStage(vector ? Kind::ExtendedA : Kind::CachedA, innerInput, BufferSlot::Work, tw, cross))
            return false;
        if (vector) {
            if (!addStage(Kind::ExtendedB, BufferSlot::Work, BufferSlot::Output, tw, dummy))
                return false;
        } else {
            auto coeff = cache(5, [&] { return CubeCoefficients(gauss, inverse); });
            if (!addStage(gauss ? Kind::Gauss : Kind::Packed, BufferSlot::Work, BufferSlot::Output, coeff, dummy))
                return false;
        }
    }
    const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
    if (trace && std::string(trace) == "1") {
        std::cerr << "A5_FFT_VARIANT variant=" << fft::a5::Name(route) << " N=" << n << " effective_batch=" << batch
                  << " launches=" << impl_->stages.size() << " workspace_bytes=" << EstimateWorkspaceSize() << '\n';
        for (const auto& stage : impl_->stages) {
            const auto& info = stage->kernel->GetKernelInfo();
            std::cerr << "A5_FFT_STAGE stage=" << stage->kernel->GetName() << " key=" << info.GetTilingId()
                      << " block_dim=" << info.GetBlockDim() << " local_memory=" << info.GetLocalMemorySize() << '\n';
        }
    }
    return true;
}

void FftC2CLargeCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    Run(input.data, output.data, stream, workspace);
}

void FftC2CLargeCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    auto* arena = static_cast<uint8_t*>(workspace.allocate(EstimateWorkspaceSize()));
    auto address = [&](BufferSlot slot) -> void* {
        switch (slot) {
            case BufferSlot::Input:
                return input;
            case BufferSlot::Output:
                return output;
            case BufferSlot::Work:
                return arena;
            case BufferSlot::Temporary:
                return arena + getAlignedSize(impl_->signalBytes);
        }
        return nullptr;
    };
    try {
        for (auto& stage : impl_->stages) {
            stage->launch.GetInTensor(0).data = address(stage->input);
            stage->launch.GetOutTensor(0).data = address(stage->output);
            stage->run.SetStream(stream);
            stage->run.SetScratchDeviceAddr(static_cast<uint8_t*>(address(stage->output)));
            if (!stage->kernel->Run(stage->launch, stage->run).Ok()) {
                throw std::runtime_error("Large C2C FFT stage launch failed.");
            }
        }
    } catch (...) {
        workspace.recycleLast();
        throw;
    }
    workspace.recycleLast();
}
