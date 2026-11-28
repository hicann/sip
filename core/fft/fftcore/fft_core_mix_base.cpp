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

#include <mki/utils/platform/platform_info.h>
#include <mki/utils/rt/rt.h>
#include "log/log.h"
#include "utils/assert.h"
#include "ops.h"
#include "fftcore/fft_core_common_func.h"
#include "utils/aspb_status.h"
#include "utils/ops_base.h"
#include "fftcore/ops_fft_kernel_stub.h"

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

    // 输出 desc 全量取 launchParam 输出张量（issue #379）：旧实现整体赋值 input.desc，
    // R2C/C2R 场景下 dims/strides 被输入（复数）描述污染；dims 为空时对 end()-1 的
    // 解引用属未定义行为，先校验非空
    if (input.desc.dims.empty() || launchParam.GetOutTensor(0).desc.dims.empty()) {
        ASDSIP_LOG(ERROR) << "FftCoreMixBase empty tensor dims, run rejected.";
        workspace.recycleLast();
        return;
    }
    output.dataSize = launchParam.GetOutTensor(0).dataSize;
    output.desc = launchParam.GetOutTensor(0).desc;

    // 下发失败抛出异常，由 exec 入口 try/catch 统一翻译为错误码（issue #568）
    if (!kernel->Run(launchParam, runInfo).Ok()) {
        ASDSIP_LOG(ERROR) << "FftCoreMixBase kernel launch failed.";
        throw std::runtime_error("FftCoreMixBase kernel launch failed.");
    }

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "FftCoreMixBase run success.";
    return;
}

void FftCoreMixBase::DestroyInDevice()
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
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
    if (opsInputIndex != nullptr) {
        MkiRtMemFreeDevice(opsInputIndex);
        opsInputIndex = nullptr;
    }
    if (opsOutputIndex != nullptr) {
        MkiRtMemFreeDevice(opsOutputIndex);
        opsOutputIndex = nullptr;
    }
    if (opsATable != nullptr) {
        MkiRtMemFreeDevice(opsATable);
        opsATable = nullptr;
    }
    if (opsBTable != nullptr) {
        MkiRtMemFreeDevice(opsBTable);
        opsBTable = nullptr;
    }
}

// stub 判空仅防护绕过统一拦截的直接调用(缺失走 internal)
void FftCoreMixBase::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (AsdSip::UseOpsFftKernelBackend() && OpsFftBackendStubReady()) {
        RunViaOpsFft(input, output, stream, workspace);
        return;
    }
    FftOperation::Run(input, output, stream, workspace);
}

bool FftCoreMixBase::OpsFftBackendStubReady() const
{
    switch (coreType) {
        case FFTCoreType::kFftMix:
            return AsdSip::OpsFftMixStubAvailable();
        case FFTCoreType::kFftR2C:
            return AsdSip::OpsFftR2CStubAvailable();
        case FFTCoreType::kFftC2R:
            return AsdSip::OpsFftC2RStubAvailable();
        default:
            return false;
    }
}

// ops-fft 直调: 常量/tiling/workspace 复用 plan 期产物;
// 变体选择对齐 ops-fft host: C2C 按 aivSplitWay(1-6), R2C/C2R 按 aivSplitWay(1-3) + 奇偶(parity)
void FftCoreMixBase::RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    // 常量首次 Run 拷贝至 device, plan 级缓存;
    // C2C 无索引/A/B 表 (nullptr tensor 跳过), R2C/C2R 全量拷贝 (奇数 N 时 A/B 表为空)
    if (!opsConstUploaded) {
        // 空常量与 ops-fft host 行为对齐: tw 单基传 nullptr; A/B 表(奇数 N 为空)传 4 字节零缓冲
        auto upload = [](void** dev, const std::shared_ptr<AsdSip::FFTensor>& tensor, bool dummyWhenEmpty) -> bool {
            if (tensor == nullptr || tensor->hostData == nullptr || tensor->dataSize == 0) {
                if (!dummyWhenEmpty) {
                    *dev = nullptr;
                    return true;
                }
                float zero = 0.0F;
                return OpsFftUploadConst(dev, &zero, sizeof(float));
            }
            return OpsFftUploadConst(dev, tensor->hostData, tensor->dataSize);
        };
        auto uploadRaw = [](void** dev, const void* host, size_t size) -> bool {
            return OpsFftUploadConst(dev, host, size);
        };
        constexpr bool K_ALLOW_NULL = false;
        constexpr bool K_DUMMY_WHEN_EMPTY = true;
        bool isX2X = (coreType == FFTCoreType::kFftR2C || coreType == FFTCoreType::kFftC2R);
        if (!upload(&opsDftMatrix, dftMatrixArray, K_DUMMY_WHEN_EMPTY) ||
            !upload(&opsRadixList, radixList, K_DUMMY_WHEN_EMPTY) ||
            !upload(&opsTwMatrix, twMatrixArray, K_ALLOW_NULL)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend(mix): dft/tw/radix upload failed";
            throw std::runtime_error("ops-fft backend(mix): dft/tw/radix upload failed");
        }
        if (isX2X &&
            (!upload(&opsInputIndex, inputIndex, K_DUMMY_WHEN_EMPTY) ||
             !upload(&opsOutputIndex, outputIndex, K_DUMMY_WHEN_EMPTY) ||
             !upload(&opsATable, aTable, K_DUMMY_WHEN_EMPTY) || !upload(&opsBTable, bTable, K_DUMMY_WHEN_EMPTY))) {
            ASDSIP_LOG(ERROR) << "ops-fft backend(mix): index/ab-table upload failed";
            throw std::runtime_error("ops-fft backend(mix): index/ab-table upload failed");
        }
        // ops-fft 上游 bbfce88(#179) 起 C2R host 系数表布局为「实部段[0,fftN) + Nyquist 端点
        // 占位 fftN + 虚部段从 fftN+1 起」(tableStride = fftN + 1), 1818dff(#322) 已将
        // C2R_even kernel 虚部读取偏移同步至该布局; sip internal kernel 与 InitABTable
        // 缓存表仍为旧布局(虚部段从 fftN 起, 无占位), 二者不可混用。C2R 直调须按
        // ops-fft 契约重排上传, 否则虚部系数错位一位, 偶数大点数 C2R 输出超差。
        if (isX2X && coreType == FFTCoreType::kFftC2R && parity == 0 && aTable != nullptr && bTable != nullptr &&
            aTable->hostData != nullptr && bTable->hostData != nullptr && aTable->dataSize > 0 &&
            bTable->dataSize > 0) {
            int64_t tableSize = static_cast<int64_t>(aTable->dataSize / (2 * sizeof(float)));
            if (tableSize > 0 && aTable->dataSize == static_cast<size_t>(2 * tableSize * sizeof(float)) &&
                bTable->dataSize == aTable->dataSize) {
                std::vector<float> hostA(static_cast<size_t>(2 * (tableSize + 1)), 0.0F);
                std::vector<float> hostB(static_cast<size_t>(2 * (tableSize + 1)), 0.0F);
                const float* srcA = static_cast<const float*>(aTable->hostData);
                const float* srcB = static_cast<const float*>(bTable->hostData);
                const int64_t stride = tableSize + 1; // Nyquist 端点占位后虚部段起点
                for (int64_t i = 0; i < tableSize; ++i) {
                    hostA[i] = srcA[i];
                    hostA[stride + i] = srcA[tableSize + i];
                    hostB[i] = srcB[i];
                    hostB[stride + i] = srcB[tableSize + i];
                }
                hostA[tableSize] = 1.0F;
                hostA[stride + tableSize] = -1.0F;
                hostB[tableSize] = 1.0F;
                hostB[stride + tableSize] = 1.0F;
                if (!uploadRaw(&opsATable, hostA.data(), hostA.size() * sizeof(float)) ||
                    !uploadRaw(&opsBTable, hostB.data(), hostB.size() * sizeof(float))) {
                    ASDSIP_LOG(ERROR) << "ops-fft backend(mix): c2r ab-table repack upload failed";
                    throw std::runtime_error("ops-fft backend(mix): c2r ab-table repack upload failed");
                }
            } else {
                ASDSIP_LOG(ERROR) << "ops-fft backend(mix): c2r ab-table size unexpected, aSize=" << aTable->dataSize
                                  << ", bSize=" << bTable->dataSize;
                throw std::runtime_error("ops-fft backend(mix): c2r ab-table size unexpected");
            }
        }
        opsConstUploaded = true;
    }

    // workspace: 与 sip kernel 同源同尺寸 (wsIn+wsOut+wsSync+wsC2c+wsAux)
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    void* scratch = workspace.allocate(bufferSize);

    // blocks=min(batch, min(coreNum, 20)), plan 级缓存
    if (opsCachedBlocks == 0) {
        uint32_t cubeCoreNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
        opsCachedBlocks = AsdSip::OpsFftBatchCoreNum(problemDesc.batch, cubeCoreNum);
    }
    uint8_t* sync = nullptr;
    if (aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync)) != 0 || sync == nullptr) {
        ASDSIP_LOG(ERROR) << "ops-fft backend: get hardware sync addr failed";
        workspace.recycleLast();
        throw std::runtime_error("ops-fft backend: get hardware sync addr failed");
    }

    uint32_t splitWay = static_cast<uint32_t>(aivSplitWay);
    if (coreType == FFTCoreType::kFftMix) {
        switch (splitWay) {
            case 1:
                fft_mix_1(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            case 2:
                fft_mix_2(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            case 3:
                fft_mix_3(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            case 4:
                fft_mix_4(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            case 5:
                fft_mix_5(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            case 6:
                fft_mix_6(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
            default:
                fft_mix_1(opsCachedBlocks, nullptr, stream, sync, input, opsDftMatrix, opsTwMatrix, opsRadixList,
                          output, scratch, runInfo.GetTilingDeviceAddr());
                break;
        }
        ASDSIP_LOG(INFO) << "FftCoreMix run via ops-fft kernel (mix" << splitWay << ", blocks=" << opsCachedBlocks
                         << ") success.";
    } else {
        // R2C/C2R: 偶数 N 用 _even 变体, 奇数 N 用 _odd 变体 (对齐 ops-fft host 变体选择)
        bool odd = (parity == 1);
        auto launchReal = [this, sync, input, output, scratch, stream](auto kernelFunc) {
            kernelFunc(opsCachedBlocks, nullptr, stream, sync, input, opsInputIndex, opsATable, opsBTable,
                       opsOutputIndex, opsDftMatrix, opsTwMatrix, opsRadixList, output, scratch,
                       runInfo.GetTilingDeviceAddr());
        };
        if (coreType == FFTCoreType::kFftR2C) {
            switch (splitWay) {
                case 2:
                    odd ? launchReal(fft_r2c_2_odd) : launchReal(fft_r2c_2_even);
                    break;
                case 3:
                    odd ? launchReal(fft_r2c_3_odd) : launchReal(fft_r2c_3_even);
                    break;
                default:
                    odd ? launchReal(fft_r2c_1_odd) : launchReal(fft_r2c_1_even);
                    break;
            }
            ASDSIP_LOG(INFO) << "FftR2CCore run via ops-fft kernel (r2c" << splitWay << (odd ? "_odd" : "_even")
                             << ", blocks=" << opsCachedBlocks << ") success.";
        } else {
            switch (splitWay) {
                case 2:
                    odd ? launchReal(fft_c2r_2_odd) : launchReal(fft_c2r_2_even);
                    break;
                case 3:
                    odd ? launchReal(fft_c2r_3_odd) : launchReal(fft_c2r_3_even);
                    break;
                default:
                    odd ? launchReal(fft_c2r_1_odd) : launchReal(fft_c2r_1_even);
                    break;
            }
            ASDSIP_LOG(INFO) << "FftC2RCore run via ops-fft kernel (c2r" << splitWay << (odd ? "_odd" : "_even")
                             << ", blocks=" << opsCachedBlocks << ") success.";
        }
    }

    workspace.recycleLast();
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
    ASDSIP_ECHECK(op->InferShape(launchParam).Ok(), "InferShape failed.", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);
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
    ASDSIP_ECHECK(kernel->Init(launchParam).Ok(), "Kernel Init failed.", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

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
