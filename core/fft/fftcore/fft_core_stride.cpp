/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include "log/log.h"
#include "utils/assert.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/ops_fft_kernel_stub.h"
#include "fft_stride.h"
#include "ops.h"
#include "fftcore/fft_core_stride.h"

using namespace AsdSip;

constexpr int N_DOING_256 = 256;
constexpr int N_DOING_512 = 512;
constexpr int N_DOING_1024 = 1024;
constexpr int N_DOING_2048 = 2048;
constexpr int N_DOING_4096 = 4096;
constexpr int N_DOING_8192 = 8192;
constexpr int N_DOING_16384 = 16384;
constexpr int N_DOING_32768 = 32768;
constexpr int N_DOING_65536 = 65536;
constexpr int N_DOING_131072 = 131072;
constexpr int N_DOING_262144 = 262144;

size_t FFTCoreStride::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FFTCoreStride::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    // set workspace
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));

    runInfo.SetStream(stream);
    launchParam.GetInTensor(0).data = input.data;
    launchParam.GetOutTensor(0).data = output.data;
    // 输出 desc 全量取 launchParam 输出张量（issue #379）：旧实现整体赋值 input.desc，
    // R2C/C2R 场景下 dtype/dims/strides 被输入（复数）描述污染；dims 为空时先校验
    if (input.desc.dims.empty() || launchParam.GetOutTensor(0).desc.dims.empty()) {
        ASDSIP_LOG(ERROR) << "FFTCoreStride empty tensor dims, run rejected.";
        workspace.recycleLast();
        return;
    }
    output.desc = launchParam.GetOutTensor(0).desc;
    // 下发失败抛出异常，由 exec 入口 try/catch 统一翻译为错误码（issue #568）
    if (!kernel->Run(launchParam, runInfo).Ok()) {
        ASDSIP_LOG(ERROR) << "FFTCoreStride kernel launch failed.";
        throw std::runtime_error("FFTCoreStride kernel launch failed.");
    }

    workspace.recycleLast();

    return;
}

// stub 判空仅防护绕过统一拦截的直接调用(缺失走 internal)
void FFTCoreStride::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (AsdSip::UseOpsFftKernelBackend() && AsdSip::OpsFftStrideStubAvailable()) {
        RunViaOpsFft(input, output, stream, workspace);
        return;
    }
    FftOperation::Run(input, output, stream, workspace);
}

bool FFTCoreStride::OpsFftBackendStubReady() const { return AsdSip::OpsFftStrideStubAvailable(); }

// ops-fft 直调: 常量/tiling 复用 plan 期产物
void FFTCoreStride::RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    // 常量首次 Run 拷贝至 device, plan 级缓存
    if (opsSMatrix == nullptr) {
        if (sMatrix == nullptr || sMatrix->hostData == nullptr || sMatrix->dataSize == 0) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: sMatrix is empty";
            throw std::runtime_error("ops-fft backend: sMatrix is empty");
        }
        if (!OpsFftUploadConst(&opsSMatrix, sMatrix->hostData, sMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: sMatrix upload failed";
            throw std::runtime_error("ops-fft backend: sMatrix upload failed");
        }
    }

    // workspace: 与 sip kernel 同源同尺寸 (n * s0 * sizeof(float) * 4)
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    void* scratch = workspace.allocate(bufferSize);

    // blocks/sync: 全核 launch, plan 级缓存
    if (opsCachedBlocks == 0) {
        uint32_t coreNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
        opsCachedBlocks = AsdSip::OpsFftAllCoreNum(coreNum);
    }
    uint8_t* sync = nullptr;
    if (aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync)) != 0 || sync == nullptr) {
        ASDSIP_LOG(ERROR) << "ops-fft backend: get hardware sync addr failed";
        workspace.recycleLast();
        throw std::runtime_error("ops-fft backend: get hardware sync addr failed");
    }

    fft_stride(opsCachedBlocks, nullptr, stream, sync, input, opsSMatrix, output, scratch,
               runInfo.GetTilingDeviceAddr());

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "FFTCoreStride run via ops-fft kernel (blocks=" << opsCachedBlocks << ") success.";
}

AspbStatus FFTCoreStride::InitTactic()
{
    SVector<size_t> sradixVec;
    for (auto x : radixVec) {
        sradixVec.push_back(x);
    }

    OpParam::FftStride param = {1, problemDesc.nDoing, problemDesc.nLeft, problemDesc.forward, sradixVec};

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1, problemDesc.nDoing, problemDesc.nLeft}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_COMPLEX64;

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*sMatrix);
    launchParam.AddOutTensor(tensorOut);

    Operation* op = Ops::Instance().GetOperationByName(std::string("FftStrideOperation"));
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    ASDSIP_ECHECK(op->InferShape(launchParam).Ok(), "InferShape failed.", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
    kernel = std::unique_ptr<Kernel>(op->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    // allocate and initialize tiling workspace
    uint8_t* deviceLaunchBuffer = nullptr;
    kernel->SetLaunchWithTiling(false);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        ASDSIP_LOG(ERROR) << "empty tiling size";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
    kernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
    ASDSIP_ECHECK(kernel->Init(launchParam).Ok(), "Kernel Init failed.", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "malloc device memory fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_LOG(ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

void FFTCoreStride::InitRadix()
{
    radixVec.clear();
    switch (problemDesc.nDoing) {
        case N_DOING_256:
            radixVec = {16, 16}; // {radix: 1iter, 2iter......}
            break;
        case N_DOING_512:
            radixVec = {16, 32};
            break;
        case N_DOING_1024:
            radixVec = {16, 64};
            break;
        case N_DOING_2048:
            radixVec = {32, 64};
            break;
        case N_DOING_4096:
            radixVec = {64, 64};
            break;
        case N_DOING_8192:
            radixVec = {16, 16, 32};
            break;
        case N_DOING_16384:
            radixVec = {16, 32, 32};
            break;
        case N_DOING_32768:
            radixVec = {32, 32, 32};
            break;
        case N_DOING_65536:
            radixVec = {32, 32, 64};
            break;
        case N_DOING_131072:
            radixVec = {32, 64, 64};
            break;
        case N_DOING_262144:
            radixVec = {64, 64, 64};
            break;
        default:
            ASDSIP_LOG(ERROR) << "FFTCoreStride fftN is not in [2^8, 2^18] or not 2^n,init_radix failed";
            throw std::runtime_error("FFTCoreStride fftN is not in [2^8, 2^18] or not 2^n,init_radix failed");
            break;
    }
}

AspbStatus FFTCoreStride::InitSMatrix()
{
    AspbStatus status = InitSMatrixCommon(coreType, radixVec, problemDesc.nDoing, problemDesc.forward, sMatrix);
    return status;
}

bool FFTCoreStride::PreAllocateInDevice()
{
    if (InitSMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    return true;
}

void FFTCoreStride::DestroyInDevice()
{
    uint8_t* deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
    if (opsSMatrix != nullptr) {
        MkiRtMemFreeDevice(opsSMatrix);
        opsSMatrix = nullptr;
    }
}
