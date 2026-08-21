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
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/run_info.h>
#include <mki/utils/SVector/SVector.h>
#include "log/log.h"
#include "utils/assert.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "fft_n.h"

#include "fftcore/fft_core_n.h"

using namespace AsdSip;

constexpr int RADIXVEC_SIZE_THREE = 3;
constexpr int N_DOING_24 = 16777216;  // pow(2, 24)
constexpr int N_DOING_27 = 134217728; // pow(2, 27)
constexpr int N_DOING_15 = 32768;     // pow(2, 15)
constexpr int N_DOING_19 = 524288;    // pow(2, 19)
constexpr int N_DOING_25 = 33554432;  // pow(2, 25)
constexpr int RADIX_INDEX2 = 2;
constexpr int RADIX_INDEX3 = 3;
constexpr int RADIX_INDEX4 = 4;
constexpr int CALCUL_TWO = 2;
constexpr int CALCUL_THREE = 3;
constexpr int CALCUL_FOUR = 4;
constexpr int CALCUL_FIVE = 5;
constexpr int LOGN_MINUS6 = 6;
constexpr int LOGN_MINUS9 = 9;
constexpr int LOGN_MINUS12 = 12;
constexpr int LOGN_MINUS15 = 15;
constexpr int LOGN_13 = 13;
constexpr int LOGN_14 = 14;
constexpr int LOGN_15 = 15;
constexpr int LOGN_16 = 16;
constexpr int LOGN_17 = 17;
constexpr int LOGN_19 = 19;
constexpr int LOGN_20 = 20;
constexpr int LOGN_21 = 21;
constexpr int LOGN_22 = 22;
constexpr int LOGN_23 = 23;
constexpr int LOGN_25 = 25;
constexpr int LOGN_26 = 26;
constexpr int LOGN_27 = 27;

size_t FFTCoreN::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FFTCoreN::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
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

    return;
}

void FFTCoreN::InitTilingArgs()
{
    int n = problemDesc.nDoing;
    int iterCount = static_cast<int>(radixVec.size());
    // initiate input and output addr of iteration, addr = 1 denote output hbm
    if (iterCount == RADIXVEC_SIZE_THREE) {
        aicInputAddr = {1, 1, 0};
        aivOutputAddr = {1, 0, 1};
    } else if (n <= N_DOING_24) {
        aicInputAddr = {1, 0, 1, 0};
        aivOutputAddr = {0, 1, 0, 1};
    } else if (n <= N_DOING_27) {
        aicInputAddr = {1, 0, 0, 1, 0};
        aivOutputAddr = {0, 0, 1, 0, 1};
    } else {
        aicInputAddr = {1, 1, 0, 1, 0};
        aivOutputAddr = {0, 0, 1, 0, 1};
    }

    lessTCopy = {1, 1, 1, 1, 1};
    syncTilingNum = {4, 4, 4, 4, 4};

    if (n <= N_DOING_15) {
        syncTilingNum = {4, 6, 6, 4, 4};
    }

    if (n == N_DOING_19) {
        aicInputAddr = {0, 1, 1, 0};
        aivOutputAddr = {1, 1, 0, 1};
        syncTilingNum = {2, 2, 2, 2, 2};
    }

    if (n == N_DOING_25) {
        lessTCopy = {0, 1, 1, 1, 1};
    }

    float batchDataSize = float(n) * 2 * 4 / 1024 / 1024;
    float l2CacheSize = 92; // 910B4 cache size
    repeatBatchSize = static_cast<uint32_t>(floor((l2CacheSize - 10 - CALCUL_TWO) / CALCUL_TWO / batchDataSize));
    repeatBatchSize = repeatBatchSize < 1 ? problemDesc.batch : repeatBatchSize;
}

AspbStatus FFTCoreN::InitTactic()
{
    SVector<size_t> sradixVec;
    SVector<size_t> saicInputAddr;
    SVector<size_t> saivOutputAddr;
    SVector<size_t> slessTCopy;
    SVector<size_t> ssyncTilingNum;
    for (auto x : radixVec) {
        sradixVec.push_back(x);
    }

    InitTilingArgs();
    for (auto x : aicInputAddr) {
        saicInputAddr.push_back(x);
    }
    for (auto x : aivOutputAddr) {
        saivOutputAddr.push_back(x);
    }
    for (auto x : lessTCopy) {
        slessTCopy.push_back(x);
    }
    for (auto x : syncTilingNum) {
        ssyncTilingNum.push_back(x);
    }
    OpParam::FftN param = {problemDesc.nDoing, problemDesc.batch, repeatBatchSize, problemDesc.forward, sradixVec,
                           saicInputAddr,      saivOutputAddr,    slessTCopy,      ssyncTilingNum};

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

    Operation* op = Ops::Instance().GetOperationByName(std::string("FftNOperation"));
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

    return AsdSip::ErrorType::ACL_SUCCESS;
}

void FFTCoreN::InitRadix()
{
    int64_t minRadix = 8;
    int64_t maxRadix = 64;

    int64_t n = problemDesc.nDoing;

    int64_t logN = static_cast<int64_t>(log(n) / log(2));

    // radix is larger on the back of vector
    if (n >= pow(minRadix, RADIX_INDEX2) && n <= pow(maxRadix, RADIX_INDEX2)) {
        radixVec = {8, 8};
        for (int64_t idx = 0; idx < logN - LOGN_MINUS6; idx++) {
            radixVec[1 - idx % CALCUL_TWO] *= CALCUL_TWO;
        }
    } else if (n >= pow(minRadix, RADIX_INDEX3) && n <= pow(maxRadix, RADIX_INDEX3)) {
        radixVec = {8, 8, 8};
        for (int64_t idx = 0; idx < logN - LOGN_MINUS9; idx++) {
            radixVec[CALCUL_TWO - idx % CALCUL_THREE] *= CALCUL_TWO;
        }
    } else if (n >= pow(minRadix, RADIX_INDEX4) && n <= pow(maxRadix, RADIX_INDEX4)) {
        radixVec = {8, 8, 8, 8};
        for (int64_t idx = 0; idx < logN - LOGN_MINUS12; idx++) {
            radixVec[CALCUL_THREE - idx % CALCUL_FOUR] *= CALCUL_TWO;
        }
    } else {
        radixVec = {8, 8, 8, 8, 8};
        for (int64_t idx = 0; idx < logN - LOGN_MINUS15; idx++) {
            radixVec[CALCUL_FOUR - idx % CALCUL_FIVE] *= CALCUL_TWO;
        }
    }

    if (logN == LOGN_13) {
        radixVec = {8, 16, 64};
    } else if (logN == LOGN_14) {
        radixVec = {16, 16, 64};
    } else if (logN == LOGN_15) {
        radixVec = {32, 16, 64};
    } else if (logN == LOGN_16) {
        radixVec = {32, 32, 64};
    } else if (logN == LOGN_17) {
        radixVec = {64, 32, 64};
    } else if (logN == LOGN_19) {
        radixVec = {2, 64, 64, 64};
    } else if (logN == LOGN_20) {
        radixVec = {32, 16, 32, 64};
    } else if (logN == LOGN_21) {
        radixVec = {16, 32, 64, 64};
    } else if (logN == LOGN_22) {
        radixVec = {16, 64, 64, 64};
    } else if (logN == LOGN_23) {
        radixVec = {64, 32, 64, 64};
    } else if (logN == LOGN_25) {
        radixVec = {32, 32, 8, 64, 64};
    } else if (logN == LOGN_26) {
        radixVec = {64, 32, 16, 32, 64};
    } else if (logN == LOGN_27) {
        radixVec = {64, 64, 16, 32, 64};
    }
}

AspbStatus FFTCoreN::InitIndex()
{
    int64_t n = problemDesc.nDoing;
    int64_t tN = 1;
    int64_t iterCount = static_cast<int64_t>(radixVec.size());
    if (iterCount > CALCUL_TWO) {
        for (int64_t it = 0; it < (iterCount - CALCUL_TWO); ++it) {
            tN *= radixVec[it];
        }
    } else {
        tN = radixVec[0];
    }

    int64_t tM = 2 * radixVec.back();
    int64_t N0 = tM < 128 ? 256 : 128;
    tN = tN < N0 ? tN : N0;
    int64_t tilingNum = (tM / 2) * tN;

    std::function<AsdSip::FFTensor*()> func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* indexMatrixPtr = new (std::nothrow) AsdSip::FFTensor;
        if (indexMatrixPtr == nullptr) {
            ASDSIP_LOG(ERROR) << "indexMatrixPtr new failed";
            throw std::runtime_error("indexMatrixPtr new failed");
        }
        AsdSip::FFTensor& indexMatrix = *indexMatrixPtr;

        indexMatrix.desc.dtype = TENSOR_DTYPE_INT32;
        indexMatrix.desc.format = TENSOR_FORMAT_ND;
        indexMatrix.desc.dims = {tilingNum};
        indexMatrix.dataSize = sizeof(int32_t) * tilingNum;

        int32_t* indexMatrixHost = nullptr;
        try {
            indexMatrixHost = new int32_t[tilingNum];
        } catch (std::bad_alloc& e) {
            delete indexMatrixPtr;
            ASDSIP_LOG(ERROR) << "indexMatrixHost alloc failed: " << e.what();
            throw std::runtime_error("indexMatrixHost alloc failed:.");
        }

        for (int64_t i = 0; i < tilingNum / 2; i++) {
            *(indexMatrixHost + 2 * i) = i * 4;
            *(indexMatrixHost + 2 * i + 1) = (i + tilingNum / 2) * 4;
        }

        indexMatrix.hostData = indexMatrixHost;

        return indexMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, 0, {n}, 0};
    index = FFTensorCache::getCoeff(key, func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus FFTCoreN::InitWMatrix()
{
    AspbStatus status = InitWMatrixCommon(coreType, radixVec, problemDesc.nDoing, problemDesc.forward, wMatrix);
    return status;
}

AspbStatus FFTCoreN::InitTMatrix()
{
    AspbStatus status = InitTMatrixCommon(coreType, radixVec, problemDesc.nDoing, tMatrix);
    return status;
}

bool FFTCoreN::PreAllocateInDevice()
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

    return true;
}

void FFTCoreN::DestroyInDevice() const
{
    uint8_t* deviceBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
}
