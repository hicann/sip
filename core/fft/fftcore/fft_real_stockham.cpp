/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fftcore/fft_real_stockham_core.h"
#include "params/fft_real_stockham.h"
#include "ops.h"
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <type_traits>

using namespace AsdSip;
using namespace Mki;

struct FftRealStockhamCore::Impl {
    OpParam::FftRealStockham param;
    OpParam::RealStockhamPlan plan;
    std::vector<std::shared_ptr<FFTensor>> coefficients;
    LaunchParam launch;
    RunInfo run;
    std::unique_ptr<Kernel> kernel;
    std::vector<uint8_t> tiling;
    Impl(unsigned n, unsigned batch, asdFftType type, bool forward)
        : param{n, batch, forward ? 0 : 1, type == asdFftType::ASCEND_FFT_C2R ? 1 : 0}, plan(param)
    {}
    ~Impl()
    {
        if (run.GetTilingDeviceAddr())
            MkiRtMemFreeDevice(run.GetTilingDeviceAddr());
    }
};

FftRealStockhamCore::FftRealStockhamCore(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch,
                                         asdFftType type, bool forward)
    : FFTCoreBase(type == asdFftType::ASCEND_FFT_C2R ? kFftC2RStockham : kFftR2CStockham, nDone, nDoing, nLeft, batch,
                  type, forward),
      impl_(std::make_unique<Impl>(nDoing, batch, type, forward))
{}
FftRealStockhamCore::~FftRealStockhamCore() = default;

size_t FftRealStockhamCore::EstimateWorkspaceSize()
{
    // The first region contains the donor's two ping-pong buffers. Reserve a
    // disjoint output for public exact-alias execution, then device copy-back.
    return getAlignedSize(2 * impl_->plan.bufferBytes) + getAlignedSize(impl_->plan.outputBytes);
}

bool FftRealStockhamCore::PreAllocateInDevice()
{
    if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 || problemDesc.nDone != 1 ||
        problemDesc.nLeft != 1)
        return false;
    auto& state = *impl_;
    const auto& p = state.param;
    const auto& d = state.plan;
    auto cache = [&](unsigned role, TensorDType dtype, const auto& build) {
        CoeffKey key = {coreType, role, {p.fftN}, problemDesc.forward};
        auto tensor = FFTensorCache::getCoeff(key, [&]() -> FFTensor* {
            const auto values = build();
            using Element = typename std::decay_t<decltype(values)>::value_type;
            auto result = std::make_unique<FFTensor>();
            result->desc = {dtype, TENSOR_FORMAT_ND, {static_cast<int64_t>(values.size())}, {}, 0};
            result->dataSize = values.size() * sizeof(Element);
            auto* host = new Element[values.size()];
            std::copy(values.begin(), values.end(), host);
            result->hostData = host;
            return result.release();
        });
        state.coefficients.push_back(tensor);
        return tensor;
    };
    Tensor x, y;
    x.desc = {p.isC2R ? TENSOR_DTYPE_COMPLEX64 : TENSOR_DTYPE_FLOAT,
              TENSOR_FORMAT_ND,
              {p.batchSize, OpParam::StockhamSignalElements(p, false)},
              {},
              0};
    y.desc = {p.isC2R ? TENSOR_DTYPE_FLOAT : TENSOR_DTYPE_COMPLEX64,
              TENSOR_FORMAT_ND,
              {p.batchSize, OpParam::StockhamSignalElements(p, true)},
              {},
              0};
    x.dataSize = d.inputBytes;
    y.dataSize = d.outputBytes;
    state.launch.SetParam(p);
    state.launch.AddInTensor(x);
    for (unsigned role : {1, 2, 3})
        state.launch.AddInTensor(
            *cache(role, TENSOR_DTYPE_FLOAT, [&] { return OpParam::BuildStockhamCoefficients(p, d, role); }));
    state.launch.AddInTensor(*cache(4, TENSOR_DTYPE_INT32, [&] { return d.radices; }));
    state.launch.AddOutTensor(y);

    auto* operation = Ops::Instance().GetOperationByName("FftRealStockhamOperation");
    if (!operation || !operation->InferShape(state.launch).Ok())
        return false;
    state.kernel.reset(operation->GetBestKernel(state.launch));
    if (!state.kernel || !state.kernel->CanSupport(state.launch))
        return false;
    state.kernel->SetLaunchWithTiling(false);
    const auto bytes = state.kernel->GetTilingSize(state.launch);
    if (!bytes)
        return false;
    state.tiling.resize(bytes, 0);
    state.kernel->SetTilingHostAddr(state.tiling.data(), bytes);
    if (!state.kernel->Init(state.launch).Ok())
        return false;
    void* deviceTiling = nullptr;
    if (MkiRtMemMallocDevice(&deviceTiling, bytes, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS)
        return false;
    state.run.SetTilingDeviceAddr(static_cast<uint8_t*>(deviceTiling));
    if (MkiRtMemCopy(deviceTiling, bytes, state.tiling.data(), bytes, MKIRT_MEMCOPY_HOST_TO_DEVICE) != MKIRT_SUCCESS)
        return false;

    const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
    if (trace && std::string(trace) == "1") {
        std::cerr << "A5_REAL_STOCKHAM_VARIANT kind=" << (p.isC2R ? "c2r" : "r2c") << " N=" << p.fftN
                  << " complex_N=" << d.complexN << " effective_batch=" << p.batchSize << " inverse=" << p.isInverse
                  << " launches=1 workspace_bytes=" << EstimateWorkspaceSize() << " radices=";
        for (auto radix : d.radices)
            std::cerr << radix << ',';
        std::cerr << " block_dim=" << state.kernel->GetKernelInfo().GetBlockDim() << '\n';
    }
    return true;
}

void FftRealStockhamCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    Run(input.data, output.data, stream, workspace);
}
void FftRealStockhamCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    auto& state = *impl_;
    const auto& d = state.plan;
    const auto x = reinterpret_cast<uintptr_t>(input), y = reinterpret_cast<uintptr_t>(output);
    if (!x || !y || x % 32 || y % 32 || (x != y && (x < y ? y - x < d.inputBytes : x - y < d.outputBytes)))
        throw std::invalid_argument("Real Stockham requires aligned disjoint buffers or exact aliasing");
    auto* arena = static_cast<uint8_t*>(workspace.allocate(EstimateWorkspaceSize()));
    auto* aliasOutput = arena + getAlignedSize(2 * d.bufferBytes);
    state.launch.GetInTensor(0).data = input;
    state.launch.GetOutTensor(0).data = input == output ? aliasOutput : output;
    state.run.SetStream(stream);
    state.run.SetScratchDeviceAddr(arena);
    try {
        if (!state.kernel->Run(state.launch, state.run).Ok())
            throw std::runtime_error("Real Stockham kernel launch failed");
        if (input == output && MkiRtMemCopyAsync(output, d.outputBytes, aliasOutput, d.outputBytes,
                                                 MKIRT_MEMCOPY_DEVICE_TO_DEVICE, stream) != MKIRT_SUCCESS)
            throw std::runtime_error("Real Stockham alias copy-back failed");
    } catch (...) {
        workspace.recycleLast();
        throw;
    }
    workspace.recycleLast();
}
