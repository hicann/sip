/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <stdexcept>
#include <vector>
#include <mki/utils/rt/rt.h>
#include "fftcore/fft_c2c_regbase_core.h"
#include "log/log.h"
#include "ops.h"
#include "params/fft_c2c_regbase.h"
#include "utils/ops_base.h"

using namespace AsdSip;
using namespace Mki;

FftC2CRegBaseCore::~FftC2CRegBaseCore()
{
    if (runInfo.GetTilingDeviceAddr() != nullptr) {
        MkiRtMemFreeDevice(runInfo.GetTilingDeviceAddr());
    }
}

size_t FftC2CRegBaseCore::EstimateWorkspaceSize()
{
    // The device kernel has separate input/output. Reserve one planner-owned
    // buffer so the public API can also execute an aliased input/output plan.
    return getAlignedSize(static_cast<size_t>(problemDesc.batch) * problemDesc.nDoing * 2 * sizeof(float));
}

bool FftC2CRegBaseCore::PreAllocateInDevice()
{
    const int64_t n = problemDesc.nDoing;
    const double sign = problemDesc.forward ? -1.0 : 1.0;
    CoeffKey key = {coreType, 0, {n}, problemDesc.forward};
    twiddles = FFTensorCache::getCoeff(key, [n, sign]() -> FFTensor* {
        auto tensor = std::make_unique<FFTensor>();
        tensor->desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2 * n}, {}, 0};
        tensor->dataSize = static_cast<size_t>(2 * n) * sizeof(float);
        auto* values = new float[2 * n]();
        tensor->hostData = values;
        constexpr double twoPi = 6.283185307179586476925286766559;
        for (int64_t length = 2; length <= n; length *= 2) {
            const int64_t half = length / 2;
            for (int64_t j = 0; j < half; ++j) {
                const double angle = sign * twoPi * j / length;
                const int64_t offset = 2 * (half - 1 + j);
                values[offset] = static_cast<float>(std::cos(angle));
                values[offset + 1] = static_cast<float>(std::sin(angle));
            }
        }
        return tensor.release();
    });

    OpParam::FftC2CRegBase param = {n, problemDesc.batch, problemDesc.forward ? 0 : 1};
    Tensor input;
    input.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, n}, {}, 0};
    input.dataSize = static_cast<size_t>(problemDesc.batch) * n * 2 * sizeof(float);
    Tensor output = input;
    launchParam.SetParam(param);
    launchParam.AddInTensor(input);
    launchParam.AddInTensor(*twiddles);
    launchParam.AddOutTensor(output);
    Operation* operation = Ops::Instance().GetOperationByName("FftC2CRegBaseOperation");
    if (operation == nullptr || !operation->InferShape(launchParam).Ok()) {
        ASDSIP_LOG(ERROR) << "Cannot initialize FftC2CRegBaseOperation.";
        return false;
    }
    kernel.reset(operation->GetBestKernel(launchParam));
    if (kernel == nullptr) {
        ASDSIP_LOG(ERROR) << "No RegBase kernel for " << param.ToString();
        return false;
    }
    kernel->SetLaunchWithTiling(false);
    const uint32_t tilingSize = kernel->GetTilingSize(launchParam);
    if (tilingSize == 0) {
        return false;
    }
    std::vector<uint8_t> hostTiling(tilingSize, 0);
    kernel->SetTilingHostAddr(hostTiling.data(), tilingSize);
    if (!kernel->Init(launchParam).Ok()) {
        ASDSIP_LOG(ERROR) << "Cannot initialize RegBase tiling.";
        return false;
    }
    void* deviceTiling = nullptr;
    if (MkiRtMemMallocDevice(&deviceTiling, tilingSize, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS) {
        return false;
    }
    if (MkiRtMemCopy(deviceTiling, tilingSize, hostTiling.data(), tilingSize, MKIRT_MEMCOPY_HOST_TO_DEVICE) !=
        MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceTiling);
        return false;
    }
    runInfo.SetTilingDeviceAddr(static_cast<uint8_t*>(deviceTiling));
    ASDSIP_LOG(INFO) << "A5 FFT plan backend=RegBase N=" << n << " batch=" << problemDesc.batch
                     << " inverse=" << param.isInverse << " workspace=" << EstimateWorkspaceSize();
    return true;
}

void FftC2CRegBaseCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    Run(input.data, output.data, stream, workspace);
}

void FftC2CRegBaseCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    const bool aliased = input == output;
    void* destination = aliased ? workspace.allocate(EstimateWorkspaceSize()) : output;
    launchParam.GetInTensor(0).data = input;
    launchParam.GetOutTensor(0).data = destination;
    runInfo.SetStream(stream);
    // MKI retains one zero-byte workspace slot before the tiling pointer.
    // The kernel never reads this placeholder; use a valid base address.
    runInfo.SetScratchDeviceAddr(static_cast<uint8_t*>(destination));
    const auto status = kernel->Run(launchParam, runInfo);
    if (!status.Ok()) {
        if (aliased) {
            workspace.recycleLast();
        }
        throw std::runtime_error("RegBase FFT kernel launch failed.");
    }
    if (aliased) {
        const size_t bytes = static_cast<size_t>(problemDesc.batch) * problemDesc.nDoing * 2 * sizeof(float);
        const int statusCopy = MkiRtMemCopyAsync(output, bytes, destination, bytes, MKIRT_MEMCOPY_DEVICE_TO_DEVICE,
                                                 stream);
        workspace.recycleLast();
        if (statusCopy != MKIRT_SUCCESS) {
            throw std::runtime_error("RegBase FFT in-place result copy failed.");
        }
    }
}
