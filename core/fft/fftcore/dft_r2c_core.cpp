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
#include "dft_r2c.h"
#include "base_inner_api.h"
#include "fftcore/dft_r2c_core.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;
constexpr size_t MATMUL_WORKSIZE = 64;

using namespace AsdSip;

size_t DftR2CCore::EstimateWorkspaceSize()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        ASDSIP_LOG(INFO) << "ASCEND_950 DftR2CCore workspace size.";
        return MATMUL_WORKSIZE;
    }
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DftR2CCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        FftOperation::Run(input, output, stream, workspace);
        ASDSIP_LOG(INFO) << "ASCEND_910B DftR2CCore run success.";
    } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        FftOperation::Run(input, output, stream, workspace);
        ASDSIP_LOG(INFO) << "ASCEND_950 DftR2CCore run success.";
    }
}

void DftR2CCore::DestroyInDevice() const
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

AspbStatus DftR2CCore::InitRotationMatrix()
{
    int64_t fftN = static_cast<int64_t>(problemDesc.nDoing);
    int64_t inSize = fftN;
    int64_t outSize = 2 * (fftN / 2 + 1);

    std::function<AsdSip::FFTensor*()> func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* rotationMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& rotationMatrix_ = *rotationMatrixPtr;

        float* rotationMatrixHost = nullptr;
        try {
            rotationMatrixHost = new float[outSize * inSize];
        } catch (std::bad_alloc& e) {
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "rotationMatrixHost alloc failed: " << e.what();
            throw std::runtime_error("rotationMatrixHost alloc failed:.");
        }

        float cosTable[fftN];
        float sinTable[fftN];
        for (int64_t i = 0; i < fftN; i++) {
            *(cosTable + i) = cos(K_2PI * i / fftN);
            *(sinTable + i) = sin(K_2PI * i / fftN);
        }
        for (int64_t i = 0; i < inSize; i++) {
            for (int64_t j = 0; j < (fftN / 2 + 1); j++) {
                *(rotationMatrixHost + i * outSize + 2 * j) = *(cosTable + (i * j) % fftN);
                *(rotationMatrixHost + i * outSize + 2 * j + 1) = (problemDesc.forward ? (-1.0) : (1.0)) *
                                                                  (*(sinTable + (i * j) % fftN));
            }
        }

        rotationMatrix_.desc = {
            AsdSip::TensorDType::TENSOR_DTYPE_FLOAT, AsdSip::TensorFormat::TENSOR_FORMAT_ND, {inSize, outSize}, {}, 0};
        rotationMatrix_.hostData = rotationMatrixHost;
        rotationMatrix_.dataSize = sizeof(float) * static_cast<size_t>(inSize * outSize);

        return rotationMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, 10, {static_cast<int64_t>(fftN)}, problemDesc.forward};
    rotationMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DftR2CCore init rotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void DftR2CCore::InitRadix() {}

bool DftR2CCore::PreAllocateInDevice()
{
    if (InitRotationMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "DftR2CCore PreAllocateInDevice success.";
    return true;
}

AspbStatus DftR2CCore::InitTactic()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        ASDSIP_LOG(INFO) << "ASCEND_950 DftR2CCore init tactic";
    }
    OpParam::DftR2C param = {problemDesc.nDoing, problemDesc.batch, 1 - int(problemDesc.forward)};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * problemDesc.nDoing *
                        GetTensorElementSize(AsdSip::TensorDType::TENSOR_DTYPE_FLOAT);

    unsigned outputN = problemDesc.nDoing / 2 + 1;
    tensorOut.dataSize = problemDesc.batch * outputN *
                         GetTensorElementSize(AsdSip::TensorDType::TENSOR_DTYPE_COMPLEX64);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*rotationMatrix);
    launchParam.AddOutTensor(tensorOut);

    Operation* op = Ops::Instance().GetOperationByName(std::string("DftR2COperation"));
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

    ASDSIP_LOG(INFO) << "DftR2CCore init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
