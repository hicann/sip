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
#include "base_inner_api.h"
#include "dft_c2r.h"

#include "fftcore/dft_c2r_core.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;
constexpr unsigned DIV_TWO = 2;
constexpr size_t MATMUL_WORKSIZE = 64;

using namespace AsdSip;
using namespace Mki;
size_t DftC2RCore::EstimateWorkspaceSize()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        ASDSIP_LOG(INFO) << "ASCEND_950 aclnnMatmul workspace size.";
        return MATMUL_WORKSIZE;
    }
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DftC2RCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        FftOperation::Run(input, output, stream, workspace);
        ASDSIP_LOG(INFO) << "ASCEND_910B DftC2RCore run success.";
    } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        FftOperation::Run(input, output, stream, workspace);
        ASDSIP_LOG(INFO) << "ASCEND_950 DftC2RCore run success.";
    }
}

void DftC2RCore::DestroyInDevice() const
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

AspbStatus DftC2RCore::InitRotationMatrix()
{
    size_t fftN = problemDesc.nDoing;
    size_t outSize = fftN;
    size_t inSize = 2 * (fftN / 2 + 1);

    std::function<AsdSip::FFTensor*()> func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* rotation_matrix_ptr = new AsdSip::FFTensor;
        AsdSip::FFTensor& rotation_matrix = *rotation_matrix_ptr;

        float* rotation_matrix_host = nullptr;
        try {
            rotation_matrix_host = new float[outSize * inSize]();
        } catch (std::bad_alloc& e) {
            delete rotation_matrix_ptr;
            ASDSIP_LOG(ERROR) << "rotation_matrix_host malloc failed: ";
            throw std::runtime_error("rotation_matrix_host malloc failed:.");
        }

        float cosTable[fftN];
        float sinTable[fftN];
        for (size_t i = 0; i < fftN; i++) {
            *(cosTable + i) = cos(K_2PI * i / fftN);
            *(sinTable + i) = sin(K_2PI * i / fftN);
        }
        for (size_t i = 0; i < fftN; i++) {
            for (size_t j = 0; j < fftN; j++) {
                if (i < (fftN / 2 + 1)) {
                    *(rotation_matrix_host + (2 * i) * outSize + j) = *(cosTable + (i * j) % fftN);
                    *(rotation_matrix_host + (2 * i + 1) * outSize + j) = (problemDesc.forward ? (1.0) : (-1.0)) *
                                                                          (*(sinTable + (i * j) % fftN));
                } else {
                    *(rotation_matrix_host + (2 * (fftN - i)) * outSize + j) += *(cosTable + (i * j) % fftN);
                    *(rotation_matrix_host + (2 * (fftN - i) + 1) * outSize +
                      j) += (problemDesc.forward ? (-1.0) : (1.0)) * (*(sinTable + (i * j) % fftN));
                }
            }
        }

        rotation_matrix.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                                Mki::TensorFormat::TENSOR_FORMAT_ND,
                                {static_cast<int64_t>(inSize), static_cast<int64_t>(outSize)},
                                {},
                                0};
        rotation_matrix.hostData = rotation_matrix_host;
        rotation_matrix.dataSize = sizeof(float) * inSize * outSize;

        return rotation_matrix_ptr;
    };

    AsdSip::CoeffKey key = {coreType, 11, {static_cast<int64_t>(fftN)}, problemDesc.forward};
    rotationMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DftC2RCore Init RotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void DftC2RCore::InitRadix() {}

bool DftC2RCore::PreAllocateInDevice()
{
    if (InitRotationMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "DftC2RCore PreAllocateInDevice success.";
    return true;
}

AspbStatus DftC2RCore::InitTactic()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        // ascend 950 initialization
        ASDSIP_LOG(INFO) << "ASCEND_950 DftC2RCore init tactic";
    }
    OpParam::DftC2R param = {problemDesc.nDoing, problemDesc.batch, 1 - int(problemDesc.forward)};

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {
        TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing / DIV_TWO + 1}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * (problemDesc.nDoing / DIV_TWO + 1) *
                        GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    tensorOut.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorOut.dataSize = problemDesc.batch * problemDesc.nDoing *
                         GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_FLOAT);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*rotationMatrix);
    launchParam.AddOutTensor(tensorOut);

    Operation* op = Ops::Instance().GetOperationByName(std::string("DftC2ROperation"));
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
    ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    uint8_t hostLaunchBuffer[launchBufferSize];
    kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
    kernel->Init(launchParam);

    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc device memory fail", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer, launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    ASDSIP_LOG(INFO) << "DftC2RCore init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
