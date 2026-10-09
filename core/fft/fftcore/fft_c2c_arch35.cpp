/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <cmath>
#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "fft_c2c_arch35.h"
#include "fftcore/fft_c2c_arch35_core.h"
#include "fftcore/ops_fft_kernel_stub.h"
#include "params/fft_c2c_arch35.h"

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

using namespace AsdSip;

namespace {
constexpr int64_t MIXED_RADIX_THRESHOLD = 1 << 14;
constexpr int64_t MULTI_CORE_BATCH_THRESHOLD = 32;

int32_t Log2PowerOfTwo(int64_t value)
{
    int32_t log = 0;
    while (value > 1) {
        value >>= 1;
        ++log;
    }
    return log;
}

bool ShouldUseMixedRadixKernel(bool radix2Only, int64_t fftN, int64_t batchSize)
{
    return !radix2Only || fftN < MIXED_RADIX_THRESHOLD || batchSize <= MULTI_CORE_BATCH_THRESHOLD;
}
} // namespace

size_t FftC2CCoreArch35::EstimateWorkspaceSize()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        int64_t singleBufferSize = static_cast<int64_t>(problemDesc.batch) * static_cast<int64_t>(problemDesc.nDoing) *
                                   2 * sizeof(float);
        int64_t workspaceSize = 2 * singleBufferSize;
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 workspace size: " << workspaceSize;
        return static_cast<size_t>(workspaceSize);
    }
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FftC2CCoreArch35::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        if (isMixedRadix) {
            // stub 判空仅防护绕过 exec 的直接调用(缺失时走 internal)
            if (AsdSip::UseOpsFftKernelBackend() && AsdSip::OpsFftArch35MixStubAvailable()) {
                RunViaOpsFft(input, output, stream, workspace);
                return;
            }
            int64_t batch = static_cast<int64_t>(problemDesc.batch);
            int64_t n = static_cast<int64_t>(problemDesc.nDoing);
            int64_t fullComplexFloats = batch * n * 2;
            int64_t workspaceBytes = 2 * fullComplexFloats * static_cast<int64_t>(sizeof(float));
            void* scratch = workspace.allocate(static_cast<size_t>(workspaceBytes));

            runInfo.SetScratchDeviceAddr(reinterpret_cast<uint8_t*>(scratch));
            runInfo.SetStream(stream);
            launchParam.GetInTensor(0).data = input;
            launchParam.GetOutTensor(0).data = output;

            if (kernel == nullptr) {
                ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 mixed-radix kernel pointer is null before Run.";
                workspace.recycleLast();
                return;
            }

            auto status = kernel->Run(launchParam, runInfo);
            if (!status.Ok()) {
                ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 mixed-radix kernel Run failed.";
                workspace.recycleLast();
                return;
            }
            workspace.recycleLast();
            ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 mixed-radix run success.";
            return;
        }

        int64_t batch = static_cast<int64_t>(problemDesc.batch);
        int64_t n = static_cast<int64_t>(problemDesc.nDoing);
        int64_t nHalf = n >> 1;
        int32_t cntStages = Log2PowerOfTwo(n);
        int32_t logNhalf = Log2PowerOfTwo(nHalf);
        int32_t isInverse = 1 - int(problemDesc.forward);

        int64_t fullComplexFloats = batch * n * 2;
        int64_t workspaceBytes = 2 * fullComplexFloats * static_cast<int64_t>(sizeof(float));
        float* tmp0 = static_cast<float*>(workspace.allocate(static_cast<size_t>(workspaceBytes)));
        float* tmp1 = tmp0 + fullComplexFloats;

        runInfo.SetScratchDeviceAddr(reinterpret_cast<uint8_t*>(tmp0));
        runInfo.SetStream(stream);

        if (stageTilingDeviceAddrs.size() < static_cast<size_t>(cntStages)) {
            ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 radix-2 stage tiling buffers are not initialized.";
            workspace.recycleLast();
            return;
        }

        for (int32_t stage = 0; stage < cntStages; ++stage) {
            bool isLast = (stage == cntStages - 1);
            int32_t len = 1 << (stage + 1);
            int32_t half = len >> 1;

            void* stageSrc = nullptr;
            void* stageDst = nullptr;
            if (stage == 0) {
                stageSrc = input;
            } else {
                stageSrc = (stage & 1) ? static_cast<void*>(tmp0) : static_cast<void*>(tmp1);
            }
            if (isLast) {
                stageDst = output;
            } else {
                stageDst = (stage & 1) ? static_cast<void*>(tmp1) : static_cast<void*>(tmp0);
            }

            FftC2CArch35StageTilingData stageTiling = {};
            stageTiling.batch = batch;
            stageTiling.n = n;
            stageTiling.totalButterflies = batch * nHalf;
            stageTiling.nHalf = static_cast<int32_t>(nHalf);
            stageTiling.len = len;
            stageTiling.half = half;
            stageTiling.logNhalf = logNhalf;
            stageTiling.logHalf = stage;
            stageTiling.twOffset = half - 1;
            stageTiling.outer = 1;
            stageTiling.isInverse = isInverse;
            stageTiling.scaleOut = 0;
            stageTiling.transpose = 0;

            uint8_t* stageTilingDeviceAddr = stageTilingDeviceAddrs[stage];
            int st = MkiRtMemCopy(stageTilingDeviceAddr, sizeof(stageTiling), &stageTiling, sizeof(stageTiling),
                                  MKIRT_MEMCOPY_HOST_TO_DEVICE);
            if (st != MKIRT_SUCCESS) {
                ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 copy stage tiling failed.";
                workspace.recycleLast();
                return;
            }
            runInfo.SetTilingDeviceAddr(stageTilingDeviceAddr);

            launchParam.GetInTensor(0).data = stageSrc;
            launchParam.GetOutTensor(0).data = stageDst;

            auto status = kernel->Run(launchParam, runInfo);
            if (!status.Ok()) {
                ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 radix-2 stage launch failed at stage " << stage;
                workspace.recycleLast();
                return;
            }
        }

        workspace.recycleLast();
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 run success.";
    }
}

void FftC2CCoreArch35::DestroyInDevice()
{
    if (!stageTilingDeviceAddrs.empty()) {
        for (uint8_t* deviceLaunchBuffer : stageTilingDeviceAddrs) {
            if (deviceLaunchBuffer != nullptr) {
                MkiRtMemFreeDevice(deviceLaunchBuffer);
            }
        }
        return;
    }

    uint8_t* deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
    if (opsDftMatrix != nullptr) {
        MkiRtMemFreeDevice(opsDftMatrix);
        opsDftMatrix = nullptr;
    }
    if (opsTwMatrix != nullptr) {
        MkiRtMemFreeDevice(opsTwMatrix);
        opsTwMatrix = nullptr;
    }
    if (opsRadixList != nullptr) {
        MkiRtMemFreeDevice(opsRadixList);
        opsRadixList = nullptr;
    }
}

bool FftC2CCoreArch35::OpsFftBackendStubReady() const { return AsdSip::OpsFftArch35MixStubAvailable(); }

// ops-fft 直调路径 (950/arch35 mixed-radix): 常量/tiling/workspace 均复用 sip 产物,
// tiling 为 InitTactic 上传的 FftAllMixTilingData device buffer, 与 ops-fft 逐字段同源
void FftC2CCoreArch35::RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    // 常量一次性上传 device (plan 级缓存, 首次 Run 触发)
    if (opsDftMatrix == nullptr || opsTwMatrix == nullptr || opsRadixList == nullptr) {
        auto upload = [](void** dev, void* host, size_t size) -> bool {
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
        if (!upload(&opsDftMatrix, dftMatrixArray->hostData, dftMatrixArray->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend(arch35): dftMatrix upload failed";
            return;
        }
        if (!upload(&opsTwMatrix, twMatrixArray->hostData, twMatrixArray->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend(arch35): twMatrix upload failed";
            return;
        }
        if (!upload(&opsRadixList, radixListTensor->hostData, radixListTensor->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend(arch35): radixList upload failed";
            return;
        }
    }

    // workspace: 与 internal mixed-radix 路径同源同尺寸 (2 * batch * n * complex * float)
    int64_t batch = static_cast<int64_t>(problemDesc.batch);
    int64_t n = static_cast<int64_t>(problemDesc.nDoing);
    int64_t workspaceBytes = 2 * (batch * n * 2) * static_cast<int64_t>(sizeof(float));
    void* scratch = workspace.allocate(static_cast<size_t>(workspaceBytes));

    // blocks 一次性缓存: arch35 mix 为 __vector__ kernel, 取 AIV 核数 (对齐 ops-fft GetCoreNumAiv)
    if (opsCachedBlocks == 0) {
        uint32_t coreNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_VECTOR);
        opsCachedBlocks = coreNum > 0 ? coreNum : 1;
    }

    fft_c2c_arch35_mix_multi_core(opsCachedBlocks, nullptr, stream, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                                  output, scratch, runInfo.GetTilingDeviceAddr());

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 run via ops-fft kernel (arch35 mix, blocks=" << opsCachedBlocks
                     << ") success.";
}

void FftC2CCoreArch35::InitRadix()
{
    plan.clear();
    int64_t tempN = static_cast<int64_t>(problemDesc.nDoing);

    if (tempN <= 1) {
        ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 nDoing must be greater than 1, got " << tempN;
        throw std::runtime_error("FftC2CCoreArch35 nDoing must be greater than 1.");
    }

    constexpr size_t numRadices = std::size(ALLOWED_RADICES);
    for (size_t i = 0; i < numRadices; i++) {
        int64_t radix = ALLOWED_RADICES[i];
        while (tempN % radix == 0) {
            tempN /= radix;
            plan.push_back({radix, tempN});
        }
    }

    if (tempN != 1) {
        ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 unsupported factor remains: " << tempN;
        throw std::runtime_error("FftC2CCoreArch35 unsupported factorization.");
    }

    bool radix2Only = true;
    for (const auto& stage : plan) {
        if (stage.radix != 2) {
            radix2Only = false;
            break;
        }
    }
    isMixedRadix = ShouldUseMixedRadixKernel(radix2Only, static_cast<int64_t>(problemDesc.nDoing),
                                             static_cast<int64_t>(problemDesc.batch));

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 init success. plan size: " << plan.size()
                     << ", isMixedRadix=" << isMixedRadix;
}

AspbStatus FftC2CCoreArch35::BuildFftPlan()
{
    if (plan.empty()) {
        ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 plan is empty, InitRadix not called?";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    double sign = problemDesc.forward ? (-1.0) : (1.0);
    size_t planLen = plan.size();
    size_t radixListSize = planLen * sizeof(int32_t);
    bool mixedRadixMode = isMixedRadix;
    unsigned coeffKeyBase = mixedRadixMode ? 110U : 100U;

    auto radixListFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* t = new AsdSip::FFTensor;
        int32_t* host = new int32_t[planLen];
        for (size_t s = 0; s < planLen; s++) {
            host[s] = static_cast<int32_t>(plan[s].radix);
        }
        t->desc = {Mki::TensorDType::TENSOR_DTYPE_INT32,
                   Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(planLen)},
                   {},
                   0};
        t->hostData = host;
        t->dataSize = radixListSize;
        return t;
    };
    AsdSip::CoeffKey radixKey = {
        coreType, coeffKeyBase, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    radixListTensor = FFTensorCache::getCoeff(radixKey, radixListFunc);

    size_t totalDftComplex = 0;
    if (mixedRadixMode) {
        for (const auto& stage : plan) {
            totalDftComplex += static_cast<size_t>(stage.radix * stage.radix);
        }
    }
    size_t totalDftFloats = mixedRadixMode ? totalDftComplex * 2 : 1;

    size_t totalTwComplex = 0;
    if (mixedRadixMode) {
        int64_t prev = 1;
        for (const auto& stage : plan) {
            totalTwComplex += static_cast<size_t>(stage.radix * prev);
            prev *= stage.radix;
        }
    } else {
        for (int64_t len = 2; len <= static_cast<int64_t>(problemDesc.nDoing); len <<= 1) {
            totalTwComplex += static_cast<size_t>(len >> 1);
        }
    }
    size_t totalTwFloats = totalTwComplex * 2;

    size_t dftDataSize = totalDftFloats * sizeof(float);

    auto dftFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* t = new AsdSip::FFTensor;
        float* host = new float[totalDftFloats]();

        if (mixedRadixMode) {
            size_t offset = 0;
            for (const auto& stage : plan) {
                int64_t radix = stage.radix;
                for (int64_t q = 0; q < radix; q++) {
                    for (int64_t p = 0; p < radix; p++) {
                        double angle = sign * K_2PI * p * q / radix;
                        host[(offset + q * radix + p) * 2] = static_cast<float>(std::cos(angle));
                        host[(offset + q * radix + p) * 2 + 1] = static_cast<float>(std::sin(angle));
                    }
                }
                offset += static_cast<size_t>(radix * radix);
            }
        }

        t->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                   Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(totalDftFloats)},
                   {},
                   0};
        t->hostData = host;
        t->dataSize = dftDataSize;
        return t;
    };

    size_t twDataSize = totalTwFloats * sizeof(float);

    auto twFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* t = new AsdSip::FFTensor;
        float* host = nullptr;
        try {
            host = new float[totalTwFloats]();
        } catch (std::bad_alloc& e) {
            delete t;
            ASDSIP_LOG(ERROR) << "twMatrixArray host malloc failed";
            throw std::runtime_error("twMatrixArray host malloc failed.");
        }

        size_t offset = 0;
        if (mixedRadixMode) {
            int64_t prev = 1;
            int64_t len = 1;
            for (const auto& stage : plan) {
                int64_t radix = stage.radix;
                len *= radix;
                for (int64_t p = 0; p < radix; p++) {
                    for (int64_t j = 0; j < prev; j++) {
                        double angle = sign * K_2PI * p * j / len;
                        host[(offset + p * prev + j) * 2] = static_cast<float>(std::cos(angle));
                        host[(offset + p * prev + j) * 2 + 1] = static_cast<float>(std::sin(angle));
                    }
                }
                offset += static_cast<size_t>(radix * prev);
                prev = len;
            }
        } else {
            for (int64_t len = 2; len <= static_cast<int64_t>(problemDesc.nDoing); len <<= 1) {
                int64_t half = len >> 1;
                for (int64_t j = 0; j < half; j++) {
                    double angle = sign * K_2PI * j / len;
                    host[(offset + j) * 2] = static_cast<float>(std::cos(angle));
                    host[(offset + j) * 2 + 1] = static_cast<float>(std::sin(angle));
                }
                offset += static_cast<size_t>(half);
            }
        }

        t->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT,
                   Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(totalTwFloats)},
                   {},
                   0};
        t->hostData = host;
        t->dataSize = twDataSize;
        return t;
    };

    AsdSip::CoeffKey dftKey = {
        coreType, coeffKeyBase + 1, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    dftMatrixArray = FFTensorCache::getCoeff(dftKey, dftFunc);

    AsdSip::CoeffKey twKey = {
        coreType, coeffKeyBase + 2, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    twMatrixArray = FFTensorCache::getCoeff(twKey, twFunc);

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 BuildFftPlan success. "
                     << "planLen=" << planLen << ", dftComplex=" << totalDftComplex
                     << ", twiddleComplex=" << totalTwComplex << ", isMixedRadix=" << mixedRadixMode;

    return AsdSip::ErrorType::ACL_SUCCESS;
}

bool FftC2CCoreArch35::PreAllocateInDevice()
{
    if (BuildFftPlan() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    return true;
}

AspbStatus FftC2CCoreArch35::InitTactic()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 init tactic";
    }

    OpParam::FftC2CArch35 param = {problemDesc.nDoing, problemDesc.batch, static_cast<int64_t>(plan.size()),
                                   1 - int(problemDesc.forward), isMixedRadix};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();
    Tensor tensorIn;
    Tensor tensorOut;
    int64_t inputN = problemDesc.nDoing;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorIn.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(inputN) *
                        GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorOut.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(inputN) *
                         GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*dftMatrixArray);
    launchParam.AddInTensor(*twMatrixArray);
    launchParam.AddInTensor(*radixListTensor);
    launchParam.AddOutTensor(tensorOut);

    Operation* op = Ops::Instance().GetOperationByName(opName);
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    op->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(op->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    uint8_t* deviceLaunchBuffer = nullptr;
    kernel->SetLaunchWithTiling(false);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
    kernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
    kernel->Init(launchParam);

    if (!isMixedRadix) {
        stageTilingDeviceAddrs.clear();
        stageTilingDeviceAddrs.reserve(plan.size());
        for (size_t stage = 0; stage < plan.size(); ++stage) {
            void* tempDevicePtr = nullptr;
            int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
            if (st != MKIRT_SUCCESS) {
                for (uint8_t* stageTilingAddr : stageTilingDeviceAddrs) {
                    MkiRtMemFreeDevice(stageTilingAddr);
                }
                stageTilingDeviceAddrs.clear();
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "malloc stage tiling device memory fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }

            deviceLaunchBuffer = static_cast<uint8_t*>(tempDevicePtr);
            st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
            if (st != MKIRT_SUCCESS) {
                MkiRtMemFreeDevice(deviceLaunchBuffer);
                for (uint8_t* stageTilingAddr : stageTilingDeviceAddrs) {
                    MkiRtMemFreeDevice(stageTilingAddr);
                }
                stageTilingDeviceAddrs.clear();
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy stage tiling to device fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }
            stageTilingDeviceAddrs.push_back(deviceLaunchBuffer);
        }
        runInfo.SetTilingDeviceAddr(stageTilingDeviceAddrs[0]);
    } else {
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
    }

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
