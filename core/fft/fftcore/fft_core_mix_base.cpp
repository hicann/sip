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

#include <mki/utils/rt/rt.h>
#include "log/log.h"
#include "utils/assert.h"
#include "ops.h"
#include "fftcore/fft_core_common_func.h"
#include "utils/aspb_status.h"
#include "utils/ops_base.h"

#include "fftcore/fft_core_mix_base.h"

using namespace AsdSip;

constexpr int32_t N_FFT_C2C_MAX = 2 * 128 * 128;
constexpr int32_t RADIXVEC_MAX = 128;

size_t FftCoreMixBase::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FftCoreMixBase::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    // set workspace
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));

    runInfo.SetStream(stream);
    launchParam.GetInTensor(0).data = input.data;
    launchParam.GetOutTensor(0).data = output.data;

    output.dataSize = launchParam.GetOutTensor(0).dataSize;
    output.desc = input.desc;
    output.desc.dtype = launchParam.GetOutTensor(0).desc.dtype;
    *(output.desc.dims.end() - 1) = *(launchParam.GetOutTensor(0).desc.dims.end() - 1);

    kernel->Run(launchParam, runInfo);

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "FftCoreMixBase run success.";
    return;
}

void FftCoreMixBase::DestroyInDevice() const
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

void FftCoreMixBase::InitRadix()
{
    radixVec.clear();
    if (nFftC2c >= N_FFT_C2C_MAX) {
        InitMixRadixLong(nFftC2c, radixVec);
    } else {
        InitMixRadixShort(nFftC2c, radixVec);
    }

    for (int64_t r : radixVec) {
        if (r > RADIXVEC_MAX) {
            ASDSIP_LOG(ERROR) << "One of the radixes is greater than 128.";
            throw std::runtime_error("One of the radixes is greater than 128.");
        }
    }
    ASDSIP_LOG(INFO) << "FftCoreMixBase init radix success.";
}

bool FftCoreMixBase::PreAllocateC2CInDevice()
{
    if (InitMixRadixList(coreType, nFftC2c, problemDesc.forward, radixVec, radixList) !=
        AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitMixRadixTwiddleMatrix(coreType, nFftC2c, problemDesc.batch, problemDesc.forward, radixVec,
                                  dftMatrixArray) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitMixRadixTWMatrix(coreType, nFftC2c, problemDesc.forward, radixVec, twMatrixArray) !=
        AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "FftCoreMixBase PreAllocateC2CInDevice success.";
    return true;
}

bool FftCoreMixBase::PreAllocateX2XInDevice()
{
    // init orders
    if (InitMixRadixList(coreType, nFftC2c, problemDesc.forward, radixVec, radixList) !=
        AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitMixRadixTwiddleMatrix(coreType, nFftC2c, problemDesc.batch, problemDesc.forward, radixVec,
                                  dftMatrixArray) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitMixRadixTWMatrix(coreType, nFftC2c, problemDesc.forward, radixVec, twMatrixArray) !=
        AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitInputOutputIndex(coreType, nFftC2c, parity, inputIndex, outputIndex) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitABTable(coreType, nFftC2c, problemDesc.forward, parity, aTable, bTable) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "FftCoreMixBase PreAllocateX2XInDevice success.";
    return true;
}

void FftCoreMixBase::InitParams()
{
    InitMixRadixParam(parity, nFftC2c, problemDesc.batch, radixVec, workspaceInputSize, workspaceOutputSize,
                      workspaceSyncSize, workspaceC2cOutputSize, workspaceAuxilSize);

    InitAiVSplitWay(coreType, nFftC2c, radixVec, aivSplitWay);
}

AspbStatus FftCoreMixBase::InitTactic()
{
    InitParams();

    InitLaunchParam();

    Operation* op = Ops::Instance().GetOperationByName(GetOpName());
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
    std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
    kernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
    kernel->Init(launchParam);

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
