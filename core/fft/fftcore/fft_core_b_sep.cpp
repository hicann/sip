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

#include <cmath>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include "utils/assert.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "fft_b.h"

#include "fftcore/fft_core_b_sep.h"

using namespace AsdSip;

constexpr int64_t RADIX_256 = 256;
constexpr int64_t RADIX_512 = 512;
constexpr int64_t RADIX_1024 = 1024;
constexpr int64_t RADIX_2048 = 2048;
constexpr int64_t RADIX_4096 = 4096;
constexpr int64_t RADIX_8192 = 8192;
constexpr int64_t RADIX_16384 = 16384;
constexpr int64_t RADIX_32768 = 32768;
constexpr int64_t RADIX_65536 = 65536;
constexpr int64_t RADIX_131072 = 131072;
constexpr int64_t RADIX_262144 = 262144;
constexpr int64_t FFT_N_LOG_LOW = 8;
constexpr int64_t FFT_N_LOG_HIGH = 30;
constexpr int64_t RADIXVEC_TWO = 2;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

namespace {
int64_t GetTMatrixColForSep(std::vector<int64_t>& radixVec, int64_t iter)
{
    int64_t col = 1;
    for (int64_t i = iter + 1; i < static_cast<int64_t>(radixVec.size()); ++i) {
        col *= radixVec[i];
    }
    return col;
}

int64_t GetAllWMatrixSizeForSep(std::vector<int64_t>& radixVec)
{
    int64_t size = 0;
    int64_t tempN;
    for (std::vector<int64_t>::iterator it = radixVec.begin(); it != radixVec.end(); ++it) {
        tempN = *it;
        size += tempN * tempN * 3; // contains real part, imag part and neg imag part
        if (size > INT64_MAX) {
            throw std::runtime_error("WMatrixSize large than INT64_MAX.");
        }
    }
    return size;
}

int64_t GetAllTMatrixSizeForSep(std::vector<int64_t>& radixVec)
{
    int64_t size = 0;
    int64_t tempRow;
    int64_t tempCol;
    for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()); ++it) {
        tempRow = radixVec[it];
        tempCol = GetTMatrixColForSep(radixVec, it);
        size += tempRow * tempCol * 2; // contains real part and imag part
        if (size > INT64_MAX) {
            throw std::runtime_error("TMatrixSize large than INT64_MAX.");
        }
    }

    return size;
}
} // namespace

size_t FFTCoreBSep::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FFTCoreBSep::Run(void* inputReal, void* inputImag, void* outputReal, void* outputImag, void* stream,
                      workspace::Workspace& workspace)
{
    const Mki::KernelInfo& kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));
    runInfo.SetStream(stream);

    launchParam.GetInTensor(0).data = inputReal;
    launchParam.GetInTensor(1).data = inputImag;
    launchParam.GetOutTensor(0).data = outputReal;
    launchParam.GetOutTensor(1).data = outputImag;
    kernel->Run(launchParam, runInfo);
    workspace.recycleLast();
}

AspbStatus FFTCoreBSep::InitWMatrix(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward)
{
    (void)forward;
    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* wMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& wMatrix_ = *wMatrixPtr;

        int64_t size = GetAllWMatrixSizeForSep(radixVec);
        float* wMatrixHost = new (std::nothrow) float[size];
        if (wMatrixHost == nullptr) {
            delete wMatrixPtr;
            ASDSIP_LOG(ERROR) << "wMatrixHost malloc failed: ";
            throw std::runtime_error("wMatrixHost malloc failed:.");
        }

        // wMatrix: w1_real, w1_imag, w2_real, w2_imag...
        int64_t currRadix;
        int64_t currSize;
        int64_t offset = 0;

        for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()); ++it) {
            currRadix = radixVec[it];
            currSize = currRadix * currRadix * 3; // real part, imag part, and neg imag part

            auto realOffset = offset;
            auto imagOffset = offset + currRadix * currRadix;
            auto negImagOffset = offset + currRadix * currRadix * 2;

            for (int64_t i = 0; i < currRadix; i++) {
                for (int64_t j = 0; j < currRadix; j++) {
                    wMatrixHost[realOffset + i * currRadix + j] = cos(-1.0 * K_2PI * (i * j) / currRadix); // real part
                    wMatrixHost[imagOffset + i * currRadix + j] = sin(-1.0 * K_2PI * (i * j) / currRadix); // imag part
                    wMatrixHost[negImagOffset + i * currRadix + j] = -1.0 * sin(-1.0 * K_2PI * (i * j) /
                                                                                currRadix); // neg imag part
                }
            }

            offset += currSize;
        }

        wMatrix_.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {size}, {}, 0};
        wMatrix_.hostData = wMatrixHost;
        wMatrix_.dataSize = sizeof(float) * size;
        return wMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, W_MARK, {n}, forward};
    wMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "FFTCoreBSep init WMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FFTCoreBSep::InitTMatrix(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward)
{
    (void)forward;
    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* tMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& tMatrix_ = *tMatrixPtr;

        int64_t eleNum = GetAllTMatrixSizeForSep(radixVec);

        tMatrix_.desc.dtype = TENSOR_DTYPE_FLOAT;
        tMatrix_.desc.format = TENSOR_FORMAT_ND;
        tMatrix_.desc.dims = {eleNum};
        tMatrix_.dataSize = sizeof(float) * eleNum;

        if (eleNum == 0) {
            throw std::runtime_error("Invalid malloc size");
        }

        float* tMatrixHost = new (std::nothrow) float[eleNum];
        if (tMatrixHost == nullptr) {
            delete tMatrixPtr;
            ASDSIP_LOG(ERROR) << "tMatrixHost malloc failed: ";
            throw std::runtime_error("tMatrixHost malloc failed:.");
        }

        int64_t currRadix;
        int64_t remain;
        int64_t currSize;
        int64_t offset = 0;

        for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()); ++it) {
            currRadix = radixVec[it];
            remain = GetTMatrixColForSep(radixVec, it);
            currSize = currRadix * remain * 2; // real part and imag part

            auto realOffset = offset;
            auto imagOffset = offset + currRadix * remain;

            for (int64_t i = 0; i < currRadix; i++) {
                for (int64_t j = 0; j < remain; j++) {
                    tMatrixHost[realOffset + remain * i + j] = cos(-1.0 * K_2PI * (i * j) /
                                                                   (currRadix * remain)); // real part
                    tMatrixHost[imagOffset + remain * i + j] = sin(-1.0 * K_2PI * (i * j) /
                                                                   (currRadix * remain)); // real part
                }
            }
            offset += currSize;
        }
        tMatrix_.hostData = tMatrixHost;
        return tMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, T_MARK, {n}, 0};
    tMatrix = FFTensorCache::getCoeff(key, func);
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FFTCoreBSep::InitTactic()
{
    SVector<size_t> sradixVec;
    for (auto x : radixVec) {
        sradixVec.push_back(x);
    }
    OpParam::FftB param = {problemDesc.nDoing, problemDesc.batch, problemDesc.forward, sradixVec};
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
    launchParam.AddInTensor(*wMatrix);
    launchParam.AddInTensor(*tMatrix);
    launchParam.AddOutTensor(tensorOutReal);
    launchParam.AddOutTensor(tensorOutImag);
    Operation* opFftBSep = Ops::Instance().GetOperationByName(std::string("FftBSepOperation"));
    if (opFftBSep == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    opFftBSep->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(opFftBSep->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
    // allocate and initialize tiling workspace
    uint8_t* buffer = nullptr;
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
    buffer = static_cast<uint8_t*>(tempDevicePtr);
    st = MkiRtMemCopy(buffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(buffer);
        buffer = nullptr;
        ASDSIP_LOG(ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(buffer);
    ASDSIP_LOG(INFO) << "FFTCoreBSep init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void FFTCoreBSep::InitRadix()
{
    radixVec.clear();
    switch (problemDesc.nDoing) {
        case RADIX_2048:
            radixVec = {32, 64};
            break;
        case RADIX_262144:
            radixVec = {64, 64, 64};
            break;
        case RADIX_131072:
            radixVec = {32, 64, 64};
            break;
        case RADIX_65536:
            radixVec = {32, 32, 64};
            break;
        case RADIX_32768:
            radixVec = {32, 32, 32};
            break;
        case RADIX_16384:
            radixVec = {64, 64};
            break;
        case RADIX_8192:
            radixVec = {64, 64};
            break;
        case RADIX_4096:
            radixVec = {64, 64};
            break;
        case RADIX_1024:
            radixVec = {16, 64};
            break;
        case RADIX_512:
            radixVec = {16, 32};
            break;
        case RADIX_256:
            radixVec = {16, 16}; // {radix: 1iter, 2iter......}
            break;
        default:
            ASDSIP_LOG(ERROR) << "FFTCoreBSep fftN is not in [2^8, 2^18] or not 2^n,init_radix failed";
            throw std::runtime_error("FFTCoreBSep fftN is not in [2^8, 2^18] or not 2^n,init_radix failed");
            break;
    }
}

bool FFTCoreBSep::PreAllocateInDevice()
{
    if (InitWMatrix(coreType, radixVec, problemDesc.nDoing, problemDesc.forward) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTMatrix(coreType, radixVec, problemDesc.nDoing, problemDesc.forward) != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    ASDSIP_LOG(INFO) << "FFTCoreBSep PreAllocateInDevice success.";
    return true;
}

void FFTCoreBSep::DestroyInDevice() const
{
    uint8_t* deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
}
