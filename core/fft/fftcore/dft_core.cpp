/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "dft.h"
#include "fftcore/dft_core.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

using namespace AsdSip;

size_t DFTCore::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DFTCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();

    // set workspace
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));

    runInfo.SetStream(stream);
    launchParam.GetInTensor(0).data = input.data;
    launchParam.GetOutTensor(0).data = output.data;
    output.desc = input.desc;
    kernel->Run(launchParam, runInfo);

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "DFTCore run success.";
    return;
}

void DFTCore::DestroyInDevice() const
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

AspbStatus DFTCore::InitRotationMatrix()
{
    int64_t fftN = static_cast<int64_t>(problemDesc.nDoing);
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;

    std::function<FFTensor*()> func = [=]() -> FFTensor* {
        FFTensor* rotationMatrixPtr = new FFTensor;
        FFTensor& rotationMatrix_ = *rotationMatrixPtr;

        float* rotationMatrixHost = nullptr;
        try {
            rotationMatrixHost = new float[outSize * inSize];
        } catch (std::bad_alloc& e) {
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "rotationMatrixHost malloc failed: ";
            throw std::runtime_error("rotationMatrixHost malloc failed:.");
        }

        float cosTable[fftN];
        float sinTable[fftN];
        for (int64_t i = 0; i < fftN; i++) {
            *(cosTable + i) = cos(K_2PI * i / fftN);
            *(sinTable + i) = sin(K_2PI * i / fftN);
        }
        for (int64_t i = 0; i < fftN; i++) {
            for (int64_t j = 0; j < fftN; j++) {
                *(rotationMatrixHost + (2 * i) * (2 * fftN) + 2 * j) = *(cosTable + (i * j) % fftN);
                *(rotationMatrixHost + (2 * i) * (2 * fftN) + 2 * j + 1) = -1 * (*(sinTable + (i * j) % fftN));
                *(rotationMatrixHost + (2 * i + 1) * (2 * fftN) + 2 * j) = 1 * (*(sinTable + (i * j) % fftN));
                *(rotationMatrixHost + (2 * i + 1) * (2 * fftN) + 2 * j + 1) = *(cosTable + (i * j) % fftN);
            }
        }

        rotationMatrix_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {outSize, inSize}, {}, 0};
        rotationMatrix_.hostData = rotationMatrixHost;
        rotationMatrix_.dataSize = sizeof(float) * static_cast<size_t>(outSize * inSize);

        return rotationMatrixPtr;
    };

    CoeffKey key = {coreType, 0, {outSize}, 0};
    rotationMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DFTCore init rotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void DFTCore::InitRadix() {}

bool DFTCore::PreAllocateInDevice()
{
    if (InitRotationMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    ASDSIP_LOG(INFO) << "DFTCore PreAllocateInDevice success.";
    return true;
}

AspbStatus DFTCore::InitTactic()
{
    OpParam::Dft param = {problemDesc.nDoing, problemDesc.batch, 1 - int(problemDesc.forward)};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_COMPLEX64;
    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    // Both Cube backends use the cached forward matrix. Inverse direction is
    // expressed by transB in tiling, without a second host-side transpose.
    launchParam.AddInTensor(*rotationMatrix);
    launchParam.AddOutTensor(tensorOut);
    Operation* op = Ops::Instance().GetOperationByName(std::string("DftOperation"));
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    op->InferShape(launchParam);
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

    uint8_t hostLaunchBuffer[launchBufferSize];
    kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
    kernel->Init(launchParam);
    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "malloc device memory fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer, launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_LOG(ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    ASDSIP_LOG(INFO) << "DFTCore init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
