/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <complex>
#include <numeric>
#include <iostream>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "utils/fft_mode.h"
#include "base_inner_api.h"
#include "utils/ops_base.h"

#include "fftcore/fft_core_istft_any.h"

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024; // unfold_grad 默认需求的框架worksapce大小
constexpr int DIM_TWO = 2;
constexpr int DIM_SLICE = 1;
constexpr int64_t COMPLEX_CODE = 16;

struct UnfoldParams {
    const SVector<int64_t> sizes;
    const int64_t dim;
    const int64_t size;
    const int64_t step;
};

// 计算out expected signal len
int64_t ComputerExpectedSliceSignalLen(struct IstftDesc& istftAnyParms)
{
    const bool center = istftAnyParms.center;
    const int64_t nFft = istftAnyParms.nFft;
    int64_t expectedOutputSignalLen = nFft + istftAnyParms.hopLengthOpt * (istftAnyParms.nFrames - 1);
    const auto lengthOpt = istftAnyParms.lengthOpt;
    const auto start = center ? nFft / 2 : 0;
    const auto end = [&]() -> int64_t {
        if (lengthOpt > 0) {
            return start + lengthOpt;
        }
        if (center) {
            return -(nFft / 2);
        }
        return expectedOutputSignalLen;
    }();
    return end - start + expectedOutputSignalLen;
}

// 计算归一化系数
float IstftFftNormalizationScale(int64_t normalization, int64_t signal)
{
    auto norm = static_cast<FftNormMode>(normalization);
    if (norm == FftNormMode::NONE) {
        return 1.0f;
    }

    const float scaleDenom = (norm == FftNormMode::BY_ROOT_N) ? std::sqrt(signal) : static_cast<float>(signal);
    return 1.0f / scaleDenom;
}

// 规归一化
void IstftApplyNormalization(Tensor& out, int64_t normalization, int64_t signal, void* stream, uint8_t* deviceBuffer)
{
    auto scale = IstftFftNormalizationScale(normalization, signal);
    const float epsilon = 1e-6f;
    if (std::abs(scale - 1.0f) > epsilon) {
        Muls(out, scale, out, stream, deviceBuffer);
    }
}

void Slice(Tensor& input, Tensor& out, int64_t dim, int64_t start, int64_t end, int64_t step, void* stream,
           uint8_t* deviceBuffer)
{
    input.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    SVector<int64_t> originalShape = input.desc.dims;
    input.desc.dims = {originalShape[0], originalShape[1] * 2};
    input.desc.strides[0] = input.desc.strides[0] * 2;

    out.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    originalShape = out.desc.dims;
    out.desc.dims = {originalShape[0], originalShape[1] * 2};

    // 计算新的的shape 和 strides
    Mki::SVector<int64_t> inputDims = input.desc.dims;

    Mki::SVector<int64_t> tmpDims(input.desc.dims.size(), 0);
    for (size_t i = 0; i < tmpDims.size(); i++) {
        tmpDims[i] = inputDims[i];
    }

    int64_t dim_value = (end - start) / step;
    if ((end - start) % step != 0) {
        dim_value++;
    }

    tmpDims[dim] = dim_value * 2;

    std::vector<int64_t> tmpStrides = input.desc.strides;
    SVector<int64_t> strides(tmpStrides.size(), 1);
    for (size_t i = 0; i < tmpStrides.size(); i++) {
        strides[i] = tmpStrides[i] * step;
    }

    AsStrided(input, out, tmpDims, strides, stream, start * strides[dim] * 2, deviceBuffer);
}

void Expand(const Tensor& input, Tensor& out, int64_t expandDim, int64_t expandDimValue, void* stream,
            uint8_t* deviceBuffer)
{
    SVector<int64_t> shape = input.desc.dims;
    std::vector<int64_t> strides = input.desc.strides;
    SVector<int64_t> expandStrides(strides.size(), 0);
    SVector<int64_t> expandShape(shape.size(), 0);

    if (shape[expandDim] != 1) {
        ASDSIP_LOG(ERROR) << "invalid param! Expanded original dim should be 1.";
        return;
    }

    // stirdeds
    for (size_t i = 0; i < strides.size(); i++) {
        expandStrides[i] = strides[i];
    }
    expandStrides[expandDim] = 0;

    // shape
    for (size_t i = 0; i < shape.size(); i++) {
        expandShape[i] = shape[i];
    }
    expandShape[expandDim] = expandDimValue;

    AsStrided(input, out, expandShape, expandStrides, stream, 0, deviceBuffer);
}

void GetRealAndImagTensor(Tensor& complexTensor, Tensor& realTensor, Tensor& imagTensor, void* stream,
                          uint8_t* deviceBuffer)
{
    SVector<int64_t> originalShape = complexTensor.desc.dims;
    std::vector<int64_t> originalStrieds = complexTensor.desc.strides;

    // view_as_real 形状 [..., 2]
    complexTensor.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    complexTensor.desc.dims.push_back(DIM_TWO);

    //  strides重新计算
    SVector<int64_t> realAndImagStrieds(originalStrieds.size(), 1);
    for (size_t i = 0; i < realAndImagStrieds.size(); i++) {
        realAndImagStrieds[i] = originalStrieds[i] * 2;
    }

    realTensor.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, originalShape, {}, 0};
    imagTensor.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, originalShape, {}, 0};

    // 实部
    AsStrided(complexTensor, realTensor, originalShape, realAndImagStrieds, stream, 0, deviceBuffer);
    // 虚部
    AsStrided(complexTensor, imagTensor, originalShape, realAndImagStrieds, stream, 1, deviceBuffer);

    // view_as_complex
    complexTensor.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    complexTensor.desc.dims.erase(complexTensor.desc.dims.end() - 1);
}

void ComplexTensor(Tensor& complexTensor, const Tensor& realTensor, const Tensor& imagTensor, void* stream,
                   uint8_t* deviceBuffer)
{
    // view_as_real 形状 [..., 2]
    complexTensor.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    complexTensor.desc.dims.push_back(DIM_TWO);

    MakeComplex(realTensor, imagTensor, complexTensor, -1, stream, deviceBuffer);

    // view_as_complex
    complexTensor.desc.dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    complexTensor.desc.dims.erase(complexTensor.desc.dims.end() - 1);
}

void UnfoldBackwardCp64(Mki::Tensor& grad, Mki::Tensor& out, struct UnfoldParams unfoldParms, void* stream,
                        uint8_t* tempCP64Buffer, uint8_t* deviceBuffer, uint8_t* unfoldGradBuffer)
{
    SVector<int64_t> dims = unfoldParms.sizes;
    int64_t dim = unfoldParms.dim;
    int64_t size = unfoldParms.size;
    int64_t step = unfoldParms.step;
    SVector<int64_t> originalShape = grad.desc.dims;
    SVector<int64_t> outShape = out.desc.dims;

    Tensor real;
    real.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                 TENSOR_FORMAT_ND,
                 originalShape,
                 {originalShape[1] * originalShape[2], originalShape[2], 1},
                 0};
    real.data = tempCP64Buffer;
    real.dataSize = grad.dataSize / 2;

    Tensor imag;
    imag.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                 TENSOR_FORMAT_ND,
                 originalShape,
                 {originalShape[1] * originalShape[2], originalShape[2], 1},
                 0};
    imag.data = tempCP64Buffer + grad.dataSize / 2;
    imag.dataSize = grad.dataSize / 2;

    Tensor realOut;
    realOut.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, outShape, {dims[1], 1}, 0};
    realOut.data = tempCP64Buffer + grad.dataSize;
    realOut.dataSize = out.dataSize / 2;

    Tensor imagOut;
    imagOut.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, outShape, {dims[1], 1}, 0};
    imagOut.data = tempCP64Buffer + grad.dataSize + out.dataSize / 2;
    imagOut.dataSize = out.dataSize / 2;

    // 虚实分离
    GetRealAndImagTensor(grad, real, imag, stream, deviceBuffer);
    // unfold
    UnfoldGrad(real, realOut, dims, dim, size, step, stream, unfoldGradBuffer);
    UnfoldGrad(imag, imagOut, dims, dim, size, step, stream, unfoldGradBuffer);
    // 虚实结合
    ComplexTensor(out, realOut, imagOut, stream, deviceBuffer);
}

size_t SliceYSize(struct IstftDesc istftAnyDesc)
{
    size_t dtypeSize = istftAnyDesc.returnComplex ? sizeof(std::complex<float>) : sizeof(float);
    return static_cast<size_t>(istftAnyDesc.channel * ComputerExpectedSliceSignalLen(istftAnyDesc)) * dtypeSize;
}

size_t SliceWindowSize(struct IstftDesc istftAnyDesc)
{
    size_t dtypeSize = istftAnyDesc.windowDtype == COMPLEX_CODE ? sizeof(std::complex<float>) : sizeof(float);
    return static_cast<size_t>(ComputerExpectedSliceSignalLen(istftAnyDesc)) * dtypeSize;
}

// 窗口包络 w^2 scratch 容量：按窗口长度 winLengthOpt 分配（与写入量 nFft*elemSize 一致，
// istft.cpp 参数校验强约束 winLength == nFft），避免按输出信号长度分配在 hop*(nFrames-1) < nFft
// 时容量不足导致段间越界写（PR !160 检视意见 HIGH）
size_t WindowSqScratchSize(struct IstftDesc istftAnyDesc)
{
    size_t dtypeSize = istftAnyDesc.windowDtype == COMPLEX_CODE ? sizeof(std::complex<float>) : sizeof(float);
    return static_cast<size_t>(istftAnyDesc.winLengthOpt) * dtypeSize;
}
} // namespace

size_t FFTCoreIstftAny::ComputerUnfoldBufferSize()
{
    const size_t expectedOutputSignalLen = static_cast<size_t>(istftAnyDesc.nFft +
                                                               istftAnyDesc.hopLengthOpt * (istftAnyDesc.nFrames - 1));
    return static_cast<size_t>(istftAnyDesc.channel) * expectedOutputSignalLen * sizeof(float) + SYS_WORKSPACE_SIZE;
}

size_t FFTCoreIstftAny::WindowExpandSize()
{
    size_t dtypeSize = istftAnyDesc.windowDtype == COMPLEX_CODE ? sizeof(std::complex<float>) : sizeof(float);
    return static_cast<size_t>(istftAnyDesc.nFrames * istftAnyDesc.nFft) * dtypeSize;
}

size_t FFTCoreIstftAny::SliceSize() { return WindowSqScratchSize(istftAnyDesc) + SliceYSize(istftAnyDesc); }

size_t FFTCoreIstftAny::TempYSize()
{
    size_t dtypeSize = istftAnyDesc.returnComplex ? sizeof(std::complex<float>) : sizeof(float);
    return static_cast<size_t>(istftAnyDesc.channel *
                               (istftAnyDesc.nFft + istftAnyDesc.hopLengthOpt * (istftAnyDesc.nFrames - 1))) *
           dtypeSize;
}

size_t FFTCoreIstftAny::TempCp64Size()
{
    const size_t expectedOutputSignalLen = static_cast<size_t>(istftAnyDesc.nFft +
                                                               istftAnyDesc.hopLengthOpt * (istftAnyDesc.nFrames - 1));
    size_t realAndImagOutputBufferSize = istftAnyDesc.channel * expectedOutputSignalLen * sizeof(float) * 2;
    size_t realAndImagInputBufferSize = istftAnyDesc.channel * istftAnyDesc.nFft * istftAnyDesc.nFrames *
                                        sizeof(std::complex<float>);
    return istftAnyDesc.returnComplex ? getAlignedSize(realAndImagOutputBufferSize + realAndImagInputBufferSize) : 0;
}

size_t FFTCoreIstftAny::EstimateWorkspaceSize()
{
    // unfold buffer size + input shape size (虚实分离需要) + 小算子 + slice * 2 + expand + ytenp size
    size_t unfoldBufferSize = getAlignedSize(ComputerUnfoldBufferSize());
    size_t tempCp64BufferSize = TempCp64Size();
    size_t expandSize = WindowExpandSize();
    size_t sliceSize = SliceSize();

    return unfoldBufferSize + getAlignedSize(ASYNC_WORKSPACE_SIZE) + getAlignedSize(tempCp64BufferSize) +
           getAlignedSize(expandSize) + getAlignedSize(sliceSize) + getAlignedSize(TempYSize());
}

void FFTCoreIstftAny::Run(Tensor& input, Tensor& window, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    // 参数准备
    int64_t nFft = istftAnyDesc.nFft;
    SVector<int64_t> inputShape = input.desc.dims; // size: (channel, n_frames, fft_size)
    int64_t nFrames = istftAnyDesc.nFrames;
    int64_t channel = istftAnyDesc.channel;
    int64_t hopeLength = istftAnyDesc.hopLengthOpt;
    int64_t expectedOutputSignalLen = nFft + hopeLength * (nFrames - 1);

    // buffer 准备
    tempCP64Buffer = (uint8_t*)workspace.allocate(TempCp64Size());
    deviceBuffer = (uint8_t*)workspace.allocate(ASYNC_WORKSPACE_SIZE);
    unfoldGradBuffer = (uint8_t*)workspace.allocate(ComputerUnfoldBufferSize());
    // ySliceBuffer 全仓无读写使用点，本次一并移除（检视意见顺带），Estimate 口径见 SliceSize()
    windowSliceeBuffer = (uint8_t*)workspace.allocate(WindowSqScratchSize(istftAnyDesc));
    windowExpandBuffer = (uint8_t*)workspace.allocate(WindowExpandSize());
    tempYBuffer = (uint8_t*)workspace.allocate(TempYSize());

    // sip fft 不支持 normalize, 需在这先ifft normalize 预处理
    FftNormMode norm = istftAnyDesc.normalized ? FftNormMode::BY_ROOT_N : FftNormMode::BY_N;
    int64_t normInt = static_cast<int64_t>(norm);
    IstftApplyNormalization(input, normInt, nFft, stream, deviceBuffer);

    Tensor yTemp = input; // size: (channel, n_frames, n_fft)yTemp

    // 窗口应用、重叠相加
    window.desc.dims = {1, 1, nFft};
    window.desc.strides = {nFft, nFft, 1};
    Mul(input, window, yTemp, stream, deviceBuffer); // size: yTemp (channel, n_frames, nFft)
    yTemp.desc.strides = {nFrames * nFft, nFft, 1};

    Tensor y;
    TensorDType dtype = istftAnyDesc.returnComplex ? Mki::TensorDType::TENSOR_DTYPE_COMPLEX64 :
                                                     Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    y.desc = {dtype,
              Mki::TensorFormat::TENSOR_FORMAT_ND,
              {channel, expectedOutputSignalLen},
              {expectedOutputSignalLen, 1},
              0};
    y.dataSize = static_cast<size_t>(channel * expectedOutputSignalLen) * GetTensorElementSize(dtype);
    y.data = tempYBuffer;

    struct UnfoldParams unfoldParms = {{inputShape[0], expectedOutputSignalLen}, 1, nFft, hopeLength};
    if (!istftAnyDesc.returnComplex) {
        UnfoldGrad(yTemp, y, {inputShape[0], expectedOutputSignalLen}, 1, nFft, hopeLength, stream, unfoldGradBuffer);
    } else {
        UnfoldBackwardCp64(yTemp, y, unfoldParms, stream, tempCP64Buffer, deviceBuffer, unfoldGradBuffer);
    }
    // 窗口包络计算：w^2 写入 workspace scratch，不回写用户 window 存储（避免同一 plan 重复执行时包络被平方）
    Tensor windowSq = window;
    windowSq.data = windowSliceeBuffer;
    windowSq.dataSize = static_cast<size_t>(nFft) * GetTensorElementSize(window.desc.dtype);
    Mul(window, window, windowSq, stream, deviceBuffer); // size: windowSq (1, 1, nFft)

    Tensor expandWin; // {1, nframes, fft}
    expandWin.desc = {
        window.desc.dtype, Mki::TensorFormat::TENSOR_FORMAT_ND, {1, nFrames, nFft}, {nFrames * nFft, nFft, 1}, 0};
    expandWin.dataSize = static_cast<size_t>(nFrames * nFft) * GetTensorElementSize(window.desc.dtype);
    expandWin.data = windowExpandBuffer;
    Expand(windowSq, expandWin, 1, nFrames, stream, deviceBuffer);

    // windowEnvelop size: (1, expected_output_signal_len)
    Tensor windowEnvelop;
    SVector<int64_t> windowEnvelopShape = {1, expectedOutputSignalLen};
    windowEnvelop.desc = {
        window.desc.dtype, Mki::TensorFormat::TENSOR_FORMAT_ND, windowEnvelopShape, {expectedOutputSignalLen, 1}, 0};
    windowEnvelop.data = input.data;
    windowEnvelop.dataSize = static_cast<size_t>(expectedOutputSignalLen) * GetTensorElementSize(window.desc.dtype);

    struct UnfoldParams unfoldParmsCp64 = {{1, expectedOutputSignalLen}, 1, nFft, hopeLength};
    if (window.desc.dtype == Mki::TensorDType::TENSOR_DTYPE_FLOAT) {
        UnfoldGrad(expandWin, windowEnvelop, {1, expectedOutputSignalLen}, 1, nFft, hopeLength, stream,
                   unfoldGradBuffer);
    } else {
        UnfoldBackwardCp64(expandWin, windowEnvelop, unfoldParmsCp64, stream, tempCP64Buffer, deviceBuffer,
                           unfoldGradBuffer);
    }

    // 中心裁剪和长度调整
    auto start = istftAnyDesc.center ? nFft / 2 : 0;
    auto end = [&]() -> int64_t {
        if (istftAnyDesc.lengthOpt > 0) {
            return start + istftAnyDesc.lengthOpt;
        }
        if (istftAnyDesc.center) {
            return -(nFft / 2);
        }
        return expectedOutputSignalLen;
    }();
    if (end < 0) {
        end += expectedOutputSignalLen;
    }

    // 归一化
    Div(y, windowEnvelop, y, stream, deviceBuffer);
    Slice(y, output, DIM_SLICE, start, end, 1, stream, deviceBuffer);

    // workspace 内存回收（与上方 allocate 次数配平：ySliceBuffer 死分配移除后为 6 次）
    workspace.recycleLast();
    workspace.recycleLast();
    workspace.recycleLast();
    workspace.recycleLast();
    workspace.recycleLast();
    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "FFTCoreAny run success.";
}

// void * tensor to mki tensor
void FFTCoreIstftAny::Run(void* input, void* window, void* output, void* stream, workspace::Workspace& workspace)
{
    Tensor inTensor;
    Tensor outTensor;
    Tensor windowTensor;
    SVector<int64_t> inputShape = {istftAnyDesc.channel, istftAnyDesc.nFrames, istftAnyDesc.nFft};
    std::vector<int64_t> inputStrides = {istftAnyDesc.nFrames * istftAnyDesc.nFft, istftAnyDesc.nFrames, 1};
    std::vector<int64_t> ouputStrides = {istftAnyDesc.outSignalLen, 1};
    SVector<int64_t> oputShape = {istftAnyDesc.channel, istftAnyDesc.outSignalLen};
    TensorDType dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    if (!istftAnyDesc.returnComplex) {
        dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    }

    size_t dtypeSize = istftAnyDesc.returnComplex ? sizeof(std::complex<float>) : sizeof(float);
    inTensor.desc = {dtype, Mki::TensorFormat::TENSOR_FORMAT_ND, inputShape, inputStrides, 0};
    inTensor.data = input;
    inTensor.dataSize = dtypeSize *
                        static_cast<size_t>(istftAnyDesc.nFrames * istftAnyDesc.nFft * istftAnyDesc.channel);

    outTensor.desc = {dtype, Mki::TensorFormat::TENSOR_FORMAT_ND, oputShape, ouputStrides, 0};
    outTensor.data = output;
    outTensor.dataSize = dtypeSize * static_cast<size_t>(istftAnyDesc.channel * istftAnyDesc.outSignalLen);

    if (istftAnyDesc.windowDtype == COMPLEX_CODE) {
        dtype = Mki::TensorDType::TENSOR_DTYPE_COMPLEX64;
    } else {
        dtype = Mki::TensorDType::TENSOR_DTYPE_FLOAT;
    }
    dtypeSize = (dtype == Mki::TensorDType::TENSOR_DTYPE_COMPLEX64) ? sizeof(std::complex<float>) : sizeof(float);
    windowTensor.desc = {dtype, Mki::TensorFormat::TENSOR_FORMAT_ND, {istftAnyDesc.winLengthOpt}, {1}, 0};
    windowTensor.data = window;
    windowTensor.dataSize = static_cast<size_t>(istftAnyDesc.winLengthOpt) * dtypeSize;

    Run(inTensor, windowTensor, outTensor, stream, workspace);
}
