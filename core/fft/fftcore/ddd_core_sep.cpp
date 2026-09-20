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

#include <mki/utils/rt/rt.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "ddd.h"

#include "fftcore/ddd_core_sep.h"


constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;


using namespace AsdSip;

size_t DddCoreSep::EstimateWorkspaceSize()
{
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

AspbStatus DddCoreSep::InitRotationMatrix(FFTCoreType coreType, int64_t fftN, std::shared_ptr<AsdSip::FFTensor> &rotMatPtr)
{
    int64_t inSize = fftN * 3;
    int64_t outSize = fftN;
    float flag = problemDesc.forward ? 1 : -1;

    // for simplicity, we construct one rotation matrix, the first half is real part, and the second half is the imag part.
    std::function<FFTensor *()> func = [=]() -> FFTensor* {
        FFTensor *rotationMatrixPtr = new FFTensor;
        FFTensor &rotationMatrix_ = *rotationMatrixPtr;

        float *dddRotationMatrixHost = nullptr;
        try {
            dddRotationMatrixHost = new float[outSize * inSize];
        } catch(std::bad_alloc& e) {
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "dddRotationMatrixHost malloc failed: ";
            throw std::runtime_error("dddRotationMatrixHost malloc failed:.");
        }

        std::vector<float> cosTable(static_cast<size_t>(fftN));
        std::vector<float> sinTable(static_cast<size_t>(fftN));
        for (int64_t i = 0; i < fftN; i++) {
            cosTable[i] = cos(flag * K_2PI * i / fftN);
            sinTable[i] = sin(flag * K_2PI * i / fftN);
        }
        // construct real part and imag part
        for (int64_t i = 0; i < fftN; i++) {
            for (int64_t j = 0; j < fftN; j++) {
                *(dddRotationMatrixHost + i * fftN + j) = cosTable[(i * j) % fftN]; // a
                *(dddRotationMatrixHost + fftN * fftN + i * fftN + j) = -1 * sinTable[(i * j) % fftN]; // b
                *(dddRotationMatrixHost + fftN * fftN * 2 + i * fftN + j) = 1 * sinTable[(i * j) % fftN]; // -b
            }
        }

        rotationMatrix_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {inSize, outSize}, {}, 0};
        rotationMatrix_.hostData = dddRotationMatrixHost;
        rotationMatrix_.dataSize = sizeof(float) * static_cast<size_t>(outSize * inSize);

        return rotationMatrixPtr;
    };

    CoeffKey key = {coreType, 0, {outSize}, problemDesc.forward};
    rotMatPtr = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DddCoreSep init rotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}


void DddCoreSep::Run(void *inputReal, void *inputImag, void *outputReal, void *outputImag, void *stream, workspace::Workspace &workspace)
{
    const Mki::KernelInfo &kernelInfo = kernel->GetKernelInfo();

    launchParam.GetInTensor(0).data = inputReal;
    launchParam.GetInTensor(1).data = inputImag;
    launchParam.GetOutTensor(0).data = outputReal;
    launchParam.GetOutTensor(1).data = outputImag;

    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t *)workspace.allocate(bufferSize));
    runInfo.SetStream(stream);

    kernel->Run(launchParam, runInfo);
    workspace.recycleLast();
}

AspbStatus DddCoreSep::InitTactic()
{
    OpParam::Ddd param = {problemDesc.batchSize, problemDesc.fftX, problemDesc.fftY, problemDesc.fftZ};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorInReal;
    Tensor tensorInImag;
    Tensor tensorOutReal;
    Tensor tensorOutImag;
    tensorInReal.desc = {
        TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batchSize, problemDesc.fftX, problemDesc.fftY, problemDesc.fftZ}, {}, 0};
    tensorInReal.dataSize = problemDesc.batchSize * problemDesc.fftX * problemDesc.fftY * problemDesc.fftZ * K_SIZE_OF_FLOAT;

    tensorInImag.desc = {
        TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batchSize, problemDesc.fftX, problemDesc.fftY, problemDesc.fftZ}, {}, 0};
    tensorInImag.dataSize = problemDesc.batchSize * problemDesc.fftX * problemDesc.fftY * problemDesc.fftZ * K_SIZE_OF_FLOAT;

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorInReal);
    launchParam.AddInTensor(tensorInImag);

    launchParam.AddInTensor(*rotationMatrixX);
    launchParam.AddInTensor(*rotationMatrixY);
    launchParam.AddInTensor(*rotationMatrixZ);

    launchParam.AddOutTensor(tensorOutReal);
    launchParam.AddOutTensor(tensorOutImag);

    Operation *dddOp = Ops::Instance().GetOperationByName(std::string("DddSepOperation"));
    if (dddOp == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    dddOp->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(dddOp->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    // allocate and initialize tiling workspace
    uint8_t *deviceBuffer = nullptr;
    kernel->SetLaunchWithTiling(false);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    uint8_t hostLaunchBuffer[launchBufferSize];
    kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
    kernel->Init(launchParam);

    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc device memory fail", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    deviceBuffer = static_cast<uint8_t *>(tempDevicePtr);
    st = MkiRtMemCopy(deviceBuffer, launchBufferSize, hostLaunchBuffer, launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceBuffer);
        deviceBuffer = nullptr;
        ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceBuffer);

    ASDSIP_LOG(INFO) << "DddCoreSep init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}


bool DddCoreSep::PreAllocateInDevice()
{
    if (InitRotationMatrix(coreType, problemDesc.fftX, rotationMatrixX) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitRotationMatrix(coreType, problemDesc.fftY, rotationMatrixY) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitRotationMatrix(coreType, problemDesc.fftZ, rotationMatrixZ) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "DddCoreSep PreAllocateInDevice success.";
    return true;
}

void DddCoreSep::DestroyInDevice() const
{
    uint8_t *deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
}
