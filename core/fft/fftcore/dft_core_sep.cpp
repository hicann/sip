/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <mki/utils/rt/rt.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "dft.h"
#include "fftcore/dft_core_sep.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;
constexpr int64_t K_SIZE_OF_FLOAT = sizeof(float);
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

using namespace AsdSip;

size_t DFTCoreSep::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DFTCoreSep::Run(void* inputReal, void* inputImag, void* outputReal, void* outputImag, void* stream,
                     workspace::Workspace& workspace)
{
    const Mki::KernelInfo& kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));

    launchParam.GetOutTensor(0).data = outputReal;
    launchParam.GetOutTensor(1).data = outputImag;
    launchParam.GetInTensor(0).data = inputReal;
    launchParam.GetInTensor(1).data = inputImag;

    runInfo.SetStream(stream);
    kernel->Run(launchParam, runInfo);
    workspace.recycleLast();
}

void DFTCoreSep::DestroyInDevice() const
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

AspbStatus DFTCoreSep::InitRotationMatrix()
{
    int64_t fftN = static_cast<int64_t>(problemDesc.nDoing);
    int64_t inSize = fftN * 3;
    int64_t outSize = fftN;
    float flag = problemDesc.forward ? 1 : -1;
    // for simplicity, we construct one rotation matrix, the first half is real part, and the second half is the imag
    // part.
    std::function<FFTensor*()> func = [=]() -> FFTensor* {
        FFTensor* rotationMatrixPtr = new FFTensor;
        FFTensor& rotationMatrix_ = *rotationMatrixPtr;

        float* rotationMatrixHost = new (std::nothrow) float[outSize * inSize];
        if (rotationMatrixHost == nullptr) {
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "rotationMatrixHost malloc failed: ";
            throw std::runtime_error("rotationMatrixHost malloc failed:.");
        }

        // 变长栈数组（VLA）在 fftN 较大时（用户 fftSize 上界 1<<27）必然栈溢出，
        // 改为堆分配并用 unique_ptr 管理生命周期（issue #118）
        std::unique_ptr<float[]> cosTable(new (std::nothrow) float[fftN]);
        std::unique_ptr<float[]> sinTable(new (std::nothrow) float[fftN]);
        if (cosTable == nullptr || sinTable == nullptr) {
            delete[] rotationMatrixHost;
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "cos/sin table malloc failed: ";
            throw std::runtime_error("cos/sin table malloc failed.");
        }
        for (int64_t i = 0; i < fftN; i++) {
            *(cosTable.get() + i) = cos(flag * K_2PI * i / fftN);
            *(sinTable.get() + i) = sin(flag * K_2PI * i / fftN);
        }

        // construct real part and imag part
        for (int64_t i = 0; i < fftN; i++) {
            for (int64_t j = 0; j < fftN; j++) {
                *(rotationMatrixHost + i * fftN + j) = *(cosTable.get() + (i * j) % fftN);                         // a
                *(rotationMatrixHost + fftN * fftN + i * fftN + j) = -1 * (*(sinTable.get() + (i * j) % fftN));    // b
                *(rotationMatrixHost + fftN * fftN * 2 + i * fftN + j) = 1 * (*(sinTable.get() + (i * j) % fftN)); // -b
            }
        }

        rotationMatrix_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {inSize, outSize}, {}, 0};
        rotationMatrix_.hostData = rotationMatrixHost;
        rotationMatrix_.dataSize = sizeof(float) * static_cast<size_t>(outSize * inSize);

        return rotationMatrixPtr;
    };

    CoeffKey key = {coreType, 0, {outSize}, problemDesc.forward};
    rotationMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DFTCoreSep init rotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void DFTCoreSep::InitRadix() {}

bool DFTCoreSep::PreAllocateInDevice()
{
    if (InitRotationMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    ASDSIP_LOG(INFO) << "DFTCoreSep PreAllocateInDevice success.";
    return true;
}

AspbStatus DFTCoreSep::InitTactic()
{
    OpParam::Dft param = {problemDesc.nDoing, problemDesc.batch, 0}; // construct rotation matrix instead.

    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorInReal;
    Tensor tensorInImag;
    Tensor tensorOutReal;
    Tensor tensorOutImag;

    tensorInReal.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorInReal.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_FLOAT;

    tensorInImag.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorInImag.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_FLOAT;

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorInReal);
    launchParam.AddInTensor(tensorInImag);
    launchParam.AddInTensor(*rotationMatrix);
    launchParam.AddOutTensor(tensorOutReal);
    launchParam.AddOutTensor(tensorOutImag);

    Operation* op = Ops::Instance().GetOperationByName(std::string("DftSepOperation"));
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    op->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(op->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    // allocate and initialize tiling workspace
    uint8_t* device = nullptr;
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
    device = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(device, launchBufferSize, hostLaunchBuffer, launchBufferSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(device);
        device = nullptr;
        ASDSIP_LOG(ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(device);

    ASDSIP_LOG(INFO) << "DFTCoreSep init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
