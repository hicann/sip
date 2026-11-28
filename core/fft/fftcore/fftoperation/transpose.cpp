/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftoperation/transpose.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include "base_inner_api.h"
#include "log/log.h"
#include "ops.h"
#include "params/fft_transpose.h"
#include "utils/ops_base.h"

using namespace AsdSip;

constexpr int DIM_TWO = 2;

namespace {

// 小算子返回码检查：底层失败时抛出异常，由 asdFftExec 系列入口的 try/catch
// 统一翻译为错误码，避免底层失败被静默吞掉后仍返回 ACL_SUCCESS（issue #567）
void CheckOpStatus(AspbStatus status, const char* opName)
{
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "fft op " << opName << " failed, status " << status << ".";
        throw std::runtime_error(std::string("fft op ") + opName + " failed, status " + std::to_string(status));
    }
}

// 兜底路径 RAII 守卫（issue #381 不变量）：调用方 input 描述被就地改写、workspace 已
// 分配后，任何异常退出（allocate 容量超限、AsStrided 失败）都必须与正常返回对称地
// 恢复描述并回收本路径分配；MarkAllocated 在 allocate 成功后登记，allocate 失败时
// 不回收、不误伤外层 pending 分配
class FallbackGuard {
public:
    explicit FallbackGuard(workspace::Workspace& ws, Mki::TensorDesc* descToRestore = nullptr,
                           const Mki::TensorDesc& backup = Mki::TensorDesc())
        : ws_{ws}, descToRestore_{descToRestore}, backup_{backup}
    {}
    ~FallbackGuard()
    {
        if (allocated_ && ws_.CanRecycle()) {
            ws_.recycleLast();
        }
        if (descToRestore_ != nullptr) {
            *descToRestore_ = backup_;
        }
    }
    void MarkAllocated() { allocated_ = true; }

private:
    workspace::Workspace& ws_;
    Mki::TensorDesc* descToRestore_ = nullptr;
    Mki::TensorDesc backup_;
    bool allocated_ = false;
};

} // namespace

Transpose::Transpose(int axis0, int axis1, Mki::SVector<int64_t> dims) : axis0_{axis0}, axis1_{axis1}, dims_{dims} {}

Transpose::~Transpose()
{
    if (runInfo.GetTilingDeviceAddr() != nullptr) {
        MkiRtMemFreeDevice(runInfo.GetTilingDeviceAddr());
    }
}

bool Transpose::init()
{
    try {
        if (TryInitTiled()) {
            return true;
        }
    } catch (const std::exception& error) {
        ASDSIP_LOG(WARN) << "Tiled transpose initialization failed: " << error.what();
    }
    // An optional fast path must not leave a partially initialized kernel active.
    kernel.reset();
    launchParam = Mki::LaunchParam{};
    ASDSIP_LOG(WARN) << "Tiled transpose unavailable; using AsStrided.";
    return true;
}

bool Transpose::TryInitTiled()
{
    using namespace Mki;
    if (kernel != nullptr)
        return true;
    if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 || dims_.size() < 2)
        return true;
    const int rank = static_cast<int>(dims_.size());
    const int a = axis0_ < 0 ? rank + axis0_ : axis0_;
    const int b = axis1_ < 0 ? rank + axis1_ : axis1_;
    if (!((a == rank - 1 && b == rank - 2) || (b == rank - 1 && a == rank - 2)))
        return true;
    OpParam::FftTranspose param;
    param.batch = 1;
    for (int i = 0; i < rank - 2; ++i) {
        if (dims_[i] <= 0 || param.batch > std::numeric_limits<int32_t>::max() / dims_[i])
            return true;
        param.batch *= dims_[i];
    }
    param.rows = dims_[rank - 2];
    param.cols = dims_[rank - 1];
    if (!param.Valid())
        return true;
    tiledBytes_ = static_cast<size_t>(param.batch * param.rows * param.cols) * 8;
    Tensor in;
    in.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {param.batch, param.rows, param.cols}, {}, 0};
    in.dataSize = tiledBytes_;
    Tensor out = in;
    out.desc.dims = {param.batch, param.cols, param.rows};
    launchParam.SetParam(param);
    launchParam.AddInTensor(in);
    launchParam.AddOutTensor(out);
    auto* op = Ops::Instance().GetOperationByName("FftTransposeOperation");
    if (op == nullptr || !op->InferShape(launchParam).Ok())
        return false;
    kernel.reset(op->GetBestKernel(launchParam));
    if (kernel == nullptr)
        return false;
    kernel->SetLaunchWithTiling(false);
    const uint32_t size = kernel->GetTilingSize(launchParam);
    if (size == 0)
        return false;
    std::vector<uint8_t> host(size, 0);
    kernel->SetTilingHostAddr(host.data(), size);
    if (!kernel->Init(launchParam).Ok())
        return false;
    void* device = nullptr;
    if (MkiRtMemMallocDevice(&device, size, MKIRT_MEM_DEFAULT) != MKIRT_SUCCESS)
        return false;
    if (MkiRtMemCopy(device, size, host.data(), size, MKIRT_MEMCOPY_HOST_TO_DEVICE) != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(device);
        return false;
    }
    runInfo.SetTilingDeviceAddr(static_cast<uint8_t*>(device));
    return true;
}

bool Transpose::RunTiled(void* input, void* output, void* stream)
{
    if (tiledBytes_ != 0) {
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("transpose requires non-null buffers");
        }
        const auto src = reinterpret_cast<uintptr_t>(input);
        const auto dst = reinterpret_cast<uintptr_t>(output);
        const auto distance = src > dst ? src - dst : dst - src;
        if (input != output && distance < tiledBytes_) {
            throw std::invalid_argument("transpose buffers partially overlap");
        }
    }
    if (kernel == nullptr || input == output)
        return false;
    launchParam.GetInTensor(0).data = input;
    launchParam.GetOutTensor(0).data = output;
    runInfo.SetStream(stream);
    runInfo.SetScratchDeviceAddr(static_cast<uint8_t*>(output));
    if (!kernel->Run(launchParam, runInfo).Ok()) {
        throw std::runtime_error("tiled transpose launch failed");
    }
    return true;
}

// only for complex64
void Transpose::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    if (input.desc.dtype == Mki::TENSOR_DTYPE_COMPLEX64 && input.desc.dims == dims_ && input.IsContiguous() &&
        input.desc.offset == 0) {
        // A zero dataSize means unspecified capacity for internal pointer-backed tensors.
        // Known undersized buffers cannot safely use either transpose implementation.
        if ((input.dataSize != 0 && input.dataSize < tiledBytes_) ||
            (output.dataSize != 0 && output.dataSize < tiledBytes_)) {
            throw std::invalid_argument("transpose buffer capacity is too small");
        }
        if (RunTiled(input.data, output.data, stream)) {
            output.desc = input.desc;
            std::swap(output.desc.dims[dims_.size() - 2], output.desc.dims[dims_.size() - 1]);
            output.desc.strides.clear();
            output.dataSize = tiledBytes_;
            return;
        }
    }
    // 兜底路径与快路径一致地要求连续且零偏移（issue #381）：非连续视图按连续
    // strides 转置会得到错误结果，显式拒绝而非静默出错
    if (!input.IsContiguous() || input.desc.offset != 0) {
        throw std::invalid_argument("transpose fallback requires a contiguous input without storage offset");
    }
    // 就地改写 input.desc（FLOAT 重解释 + 追加末维）前先备份；守卫保证包括异常退出
    // 在内的所有路径都恢复描述并回收分配，避免调用方 Tensor 描述被永久污染（issue #381）
    const Mki::TensorDesc inputDescBackup = input.desc;
    input.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    input.desc.dims.push_back(DIM_TWO);
    FallbackGuard fallbackGuard(workspace, &input.desc, inputDescBackup);

    int axis0 = axis0_;
    if (axis0_ < 0) {
        axis0 = static_cast<int>(input.desc.dims.size()) - 1 + axis0_;
    }
    int axis1 = axis1_;
    if (axis1_ < 0) {
        axis1 = static_cast<int>(input.desc.dims.size()) - 1 + axis1_;
    }

    uint8_t* deviceBuffer = (uint8_t*)workspace.allocate(ASYNC_WORKSPACE_SIZE);
    fallbackGuard.MarkAllocated();

    Mki::SVector<int64_t> shapes = input.desc.dims;
    Mki::SVector<int64_t> strides(shapes.size(), 1);
    for (int64_t i = static_cast<int64_t>(shapes.size()) - 2; i >= 0; i--) {
        strides[i] = shapes[i + 1] * strides[i + 1];
    }
    int64_t shapeTmp = shapes[axis0];
    int64_t stridesTmp = strides[axis0];
    shapes[axis0] = shapes[axis1];
    strides[axis0] = strides[axis1];
    shapes[axis1] = shapeTmp;
    strides[axis1] = stridesTmp;
    CheckOpStatus(AsStrided(input, output, shapes, strides, stream, 0, deviceBuffer), "AsStrided");

    output.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    output.desc.dims.erase(output.desc.dims.end() - 1);
}

size_t Transpose::EstimateWorkspaceSize() { return getAlignedSize(ASYNC_WORKSPACE_SIZE); }

void Transpose::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (RunTiled(input, output, stream))
        return;
    Mki::SVector<int64_t> tmpDims = dims_;
    tmpDims.push_back(DIM_TWO);

    int axis0 = axis0_;
    if (axis0_ < 0) {
        axis0 = static_cast<int>(tmpDims.size()) - 1 + axis0_;
    }
    int axis1 = axis1_;
    if (axis1_ < 0) {
        axis1 = static_cast<int>(tmpDims.size()) - 1 + axis1_;
    }

    // 异常安全守卫：AsStrided 失败等异常退出路径，析构统一回收本路径 workspace 分配
    FallbackGuard fallbackGuard(workspace);
    uint8_t* deviceBuffer = (uint8_t*)workspace.allocate(ASYNC_WORKSPACE_SIZE);
    fallbackGuard.MarkAllocated();

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, tmpDims, {}, 0};
    tensorIn.data = input;
    tensorOut.data = output;

    Mki::SVector<int64_t> strides(tmpDims.size(), 1);
    for (int64_t i = static_cast<int64_t>(tmpDims.size()) - 2; i >= 0; i--) {
        strides[i] = tmpDims[i + 1] * strides[i + 1];
    }
    int64_t shapeTmp = tmpDims[axis0];
    int64_t stridesTmp = strides[axis0];
    tmpDims[axis0] = tmpDims[axis1];
    strides[axis0] = strides[axis1];
    tmpDims[axis1] = shapeTmp;
    strides[axis1] = stridesTmp;
    CheckOpStatus(AsStrided(tensorIn, tensorOut, tmpDims, strides, stream, 0, deviceBuffer), "AsStrided");
}
