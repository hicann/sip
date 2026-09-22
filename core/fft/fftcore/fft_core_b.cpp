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

#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include "utils/assert.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "fftcore/ops_fft_kernel_stub.h"
#include "fft_b.h"

#include "fftcore/fft_core_b.h"

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

size_t FFTCoreB::EstimateWorkspaceSize()
{
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FFTCoreB::Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace)
{
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    // set workspace
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t *)workspace.allocate(bufferSize));

    runInfo.SetStream(stream);
    launchParam.GetInTensor(0).data = input.data;
    launchParam.GetOutTensor(0).data = output.data;
    output.desc = input.desc;
    kernel->Run(launchParam, runInfo);

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "FFTCoreB run success.";
    return;
}

// asdFftExecV2 实际调用的重载; stub 判空仅防护绕过 exec 的直接调用(缺失时走 internal)
void FFTCoreB::Run(void *input, void *output, void *stream, workspace::Workspace &workspace)
{
    if (AsdSip::UseOpsFftKernelBackend() && AsdSip::OpsFftBStubAvailable()) {
        RunViaOpsFft(input, output, stream, workspace);
        return;
    }
    FftOperation::Run(input, output, stream, workspace);
}

bool FFTCoreB::OpsFftBackendStubReady() const
{
    return AsdSip::OpsFftBStubAvailable();
}

// ops-fft kernel 直调后端: 常量/tiling 复用 sip plan 期产物, exec 仅一次 stub 调用
void FFTCoreB::RunViaOpsFft(void *input, void *output, void *stream, workspace::Workspace &workspace)
{
    // 常量一次性上传 device (plan 级缓存, 首次 Run 触发)
    if (opsWMatrix == nullptr || opsTMatrix == nullptr || opsIndex == nullptr) {
        auto upload = [](void **dev, void *host, size_t size) -> bool {
            if (dev == nullptr || host == nullptr || size == 0) {
                return false;
            }
            int st = MkiRtMemMallocDevice(dev, size, MKIRT_MEM_DEFAULT);
            if (st != MKIRT_SUCCESS) {
                return false;
            }
            st = MkiRtMemCopy(*dev, size, host, size, MKIRT_MEMCOPY_HOST_TO_DEVICE);
            return st == MKIRT_SUCCESS;
        };
        if (!upload(&opsWMatrix, wMatrix->hostData, wMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: wMatrix upload failed";
            return;
        }
        if (!upload(&opsTMatrix, tMatrix->hostData, tMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: tMatrix upload failed";
            return;
        }
        if (!upload(&opsIndex, index->hostData, index->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: index upload failed";
            return;
        }
    }

    // workspace: 与 sip kernel 同源同尺寸 (SCRATCH_SIZES * needCoreNum)
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    void *scratch = workspace.allocate(bufferSize);

    // blocks / sync addr 一次性缓存 (plan 级)
    if (opsCachedBlocks == 0) {
        uint32_t maxCore = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
        uint32_t needCoreNum = static_cast<uint32_t>(problemDesc.batch) > maxCore
                                   ? maxCore : static_cast<uint32_t>(problemDesc.batch);
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        opsCachedBlocks = needCoreNum;
        if (aclrtGetHardwareSyncAddr(reinterpret_cast<void **>(&opsCachedSync)) != 0 || opsCachedSync == nullptr) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: get hardware sync addr failed";
            return;
        }
    }

    fft_b(opsCachedBlocks, nullptr, stream, opsCachedSync, input, opsWMatrix, opsTMatrix,
          opsIndex, output, scratch, runInfo.GetTilingDeviceAddr());

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "FFTCoreB run via ops-fft kernel (blocks=" << opsCachedBlocks << ") success.";
}

AspbStatus FFTCoreB::InitTactic()
{
    SVector<size_t> sradixVec;
    for (auto x : radixVec) {
        sradixVec.push_back(x);
    }
    OpParam::FftB param = {problemDesc.nDoing, problemDesc.batch, problemDesc.forward, sradixVec};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();
    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_COMPLEX64;

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*wMatrix);
    launchParam.AddInTensor(*tMatrix);
    launchParam.AddInTensor(*index);
    launchParam.AddOutTensor(tensorOut);
    Operation *op = Ops::Instance().GetOperationByName(std::string("FftBOperation"));
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    op->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(op->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    // allocate and initialize tiling workspace
    uint8_t *deviceLaunchBuffer = nullptr;
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
    deviceLaunchBuffer = static_cast<uint8_t *>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer, launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_LOG(ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    ASDSIP_LOG(INFO) << "FFTCoreB init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void FFTCoreB::InitRadix()
{
    radixVec.clear();
    switch (problemDesc.nDoing) {
        case RADIX_256:
            radixVec = {16, 16};  // {radix: 1iter, 2iter......}
            break;
        case RADIX_512:
            radixVec = {16, 32};
            break;
        case RADIX_1024:
            radixVec = {16, 64};
            break;
        case RADIX_2048:
            radixVec = {32, 64};
            break;
        case RADIX_4096:
            radixVec = {64, 64};
            break;
        case RADIX_8192:
            radixVec = {64, 64};
            break;
        case RADIX_16384:
            radixVec = {64, 64};
            break;
        case RADIX_32768:
            radixVec = {32, 32, 32};
            break;
        case RADIX_65536:
            radixVec = {32, 32, 64};
            break;
        case RADIX_131072:
            radixVec = {32, 64, 64};
            break;
        case RADIX_262144:
            radixVec = {64, 64, 64};
            break;
        default:
            ASDSIP_LOG(ERROR) << "FFTCoreB fftN is not in [2^8, 2^18] or not 2^n,init_radix failed";
            throw std::runtime_error("FFTCoreB fftN is not in [2^8, 2^18] or not 2^n,init_radix failed");
            break;
    }
}

AspbStatus FFTCoreB::InitIndex()
{
    int64_t n = problemDesc.nDoing;
    std::vector<int64_t> &radixVecRef = radixVec;

    std::function<AsdSip::FFTensor *()> func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor *indexMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor &indexMatrix = *indexMatrixPtr;

        indexMatrix.desc.dtype = TENSOR_DTYPE_INT32;
        indexMatrix.desc.format = TENSOR_FORMAT_ND;
        indexMatrix.desc.dims = {128 * 128};
        indexMatrix.dataSize = sizeof(int32_t) * 128 * 128;

        if (radixVecRef[0] == 0 || radixVecRef[1] == 0) {
            ASDSIP_LOG(ERROR) << "FFTCore2 fradixVecRef[0] == 0 || radixVecRef[1] == 0,InitIndex failed";
            throw std::runtime_error("FFTCore2 fradixVecRef[0] == 0 || radixVecRef[1] == 0,InitIndex failed");
            return nullptr;
        }

        int32_t *indexMatrixHost = new(std::nothrow) int32_t[128 * 128];
        if (indexMatrixHost == nullptr) {
            delete indexMatrixPtr;
            ASDSIP_LOG(ERROR) << "indexMatrixHost malloc failed: ";
            throw std::runtime_error("indexMatrixHost malloc failed:.");
        }

        if (n > 16384) {
            for (int64_t i = 0; i < 128 * 64; ++i) {
                *(indexMatrixHost + i * 2) = i;
                *(indexMatrixHost + i * 2 + 1) = i;
            }
        } else if (n == 8192 || n == 16384) {
            for (int64_t i = 0; i < 128 * 64; ++i) {
                *(indexMatrixHost + i * 2) = i * 4;
                *(indexMatrixHost + i * 2 + 1) = (64 * 64 + i) * 4;
            }
        } else {
            int64_t nRadix = radixVecRef[0];
            int64_t M = radixVecRef[1] * 2;
            if (M == 0) {
                M = 1;
            }
            int64_t col = 128 * 128 / M;
            int64_t fftN = nRadix * M / 2;
            if (radixVecRef[0] == radixVecRef[1] && radixVecRef[1] == 16) {
                col = col / 2;
            }
            for (int64_t batchId = 0; batchId < (col / nRadix); batchId++) {  // 总共虚实8192个数, 4096次循环
                for (int64_t i = 0; i < (M / 2 / 2); i++) {  // 每个batch的列方向, 第一个2为虚实, 第二个2为两个AIV
                    for (int64_t j = 0; j < nRadix; j++) {  // 每个batch的行方向, 即nRadix
                        *(reinterpret_cast<int32_t *>(indexMatrixHost) + batchId * fftN + (i * nRadix + j) * 2) =
                            (batchId * nRadix + i * col + j) * 4;
                        *(reinterpret_cast<int32_t *>(indexMatrixHost) + batchId * fftN + (i * nRadix + j) * 2 + 1) =
                            (M / 4 * col + batchId * nRadix + i * col + j) * 4;
                    }
                }
            }
        }

        indexMatrix.hostData = indexMatrixHost;

        return indexMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, 0, {n}, 0};
    index = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "FFTCoreB init index success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FFTCoreB::InitWMatrix()
{
    AspbStatus status = InitWMatrixCommon(coreType, radixVec, problemDesc.nDoing, problemDesc.forward, wMatrix);
    return status;
}

AspbStatus FFTCoreB::InitTMatrix()
{
    AspbStatus status = InitTMatrixCommon(coreType, radixVec, problemDesc.nDoing, tMatrix);
    return status;
}

bool FFTCoreB::PreAllocateInDevice()
{
    if (InitIndex() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitWMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    ASDSIP_LOG(INFO) << "FFTCoreB PreAllocateInDevice success.";
    return true;
}

void FFTCoreB::DestroyInDevice()
{
    uint8_t *deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
    if (opsWMatrix != nullptr) {
        MkiRtMemFreeDevice(opsWMatrix);
        opsWMatrix = nullptr;
    }
    if (opsTMatrix != nullptr) {
        MkiRtMemFreeDevice(opsTMatrix);
        opsTMatrix = nullptr;
    }
    if (opsIndex != nullptr) {
        MkiRtMemFreeDevice(opsIndex);
        opsIndex = nullptr;
    }
}
