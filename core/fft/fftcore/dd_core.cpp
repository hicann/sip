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
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/ops_fft_kernel_stub.h"
#include "dd.h"

#include "fftcore/dd_core.h"

using namespace AsdSip;

size_t DdCore::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DdCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
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
        ASDSIP_LOG(ERROR) << "DdCore empty tensor dims, run rejected.";
        workspace.recycleLast();
        return;
    }
    output.desc = launchParam.GetOutTensor(0).desc;
    // 下发失败抛出异常，由 exec 入口 try/catch 统一翻译为错误码（issue #568）
    if (!kernel->Run(launchParam, runInfo).Ok()) {
        ASDSIP_LOG(ERROR) << "DdCore kernel launch failed.";
        throw std::runtime_error("DdCore kernel launch failed.");
    }

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "DdCore run success.";
    return;
}

// stub 判空仅防护绕过统一拦截的直接调用(缺失走 internal)
void DdCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (AsdSip::UseOpsFftKernelBackend() && AsdSip::OpsFftDdStubAvailable()) {
        RunViaOpsFft(input, output, stream, workspace);
        return;
    }
    FftOperation::Run(input, output, stream, workspace);
}

bool DdCore::OpsFftBackendStubReady() const { return AsdSip::OpsFftDdStubAvailable(); }

// ops-fft 直调: 常量/tiling 复用 plan 期产物
void DdCore::RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    // 常量首次 Run 拷贝至 device, plan 级缓存
    if (opsPMatrix == nullptr || opsQMatrix == nullptr) {
        if (!OpsFftUploadConst(&opsPMatrix, pMatrix->hostData, pMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: pMatrix upload failed";
            throw std::runtime_error("ops-fft backend: pMatrix upload failed");
        }
        if (!OpsFftUploadConst(&opsQMatrix, qMatrix->hostData, qMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: qMatrix upload failed";
            throw std::runtime_error("ops-fft backend: qMatrix upload failed");
        }
    }

    // workspace: 与 sip kernel 同源同尺寸 (batch * fftX * fftY * complex64)
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

    dd(opsCachedBlocks, nullptr, stream, sync, input, opsPMatrix, opsQMatrix, output, scratch,
       runInfo.GetTilingDeviceAddr());

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "DdCore run via ops-fft kernel (blocks=" << opsCachedBlocks << ") success.";
}

AspbStatus DdCore::InitTactic()
{
    OpParam::Dd param = {problemDesc.batchSize, problemDesc.fftX, problemDesc.fftY};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {
        TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batchSize, problemDesc.fftX, problemDesc.fftY}, {}, 0};
    tensorIn.dataSize = problemDesc.batchSize * problemDesc.fftX * problemDesc.fftY * K_SIZE_OF_COMPLEX64;

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*pMatrix);
    launchParam.AddInTensor(*qMatrix);
    launchParam.AddOutTensor(tensorOut);

    Operation* op = Ops::Instance().GetOperationByName(std::string("DdOperation"));
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
    ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
    kernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
    ASDSIP_ECHECK(kernel->Init(launchParam).Ok(), "Kernel Init failed.", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc device memory fail", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    ASDSIP_LOG(INFO) << "DdCore init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus DdCore::InitPMatrix()
{
    AspbStatus status = InitPQMatrix(coreType, problemDesc.fftX, problemDesc.forward, true, pMatrix);
    return status;
}

AspbStatus DdCore::InitQMatrix()
{
    AspbStatus status = InitPQMatrix(coreType, problemDesc.fftY, problemDesc.forward, false, qMatrix);
    return status;
}

bool DdCore::PreAllocateInDevice()
{
    if (InitPMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitQMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "DdCore PreAllocateInDevice success.";
    return true;
}

void DdCore::DestroyInDevice()
{
    uint8_t* deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
    if (opsPMatrix != nullptr) {
        MkiRtMemFreeDevice(opsPMatrix);
        opsPMatrix = nullptr;
    }
    if (opsQMatrix != nullptr) {
        MkiRtMemFreeDevice(opsQMatrix);
        opsQMatrix = nullptr;
    }
}
