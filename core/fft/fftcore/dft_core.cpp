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
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "utils/aspb_status.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "fftcore/ops_fft_kernel_stub.h"
#include "dft.h"
#include "fftcore/dft_core.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;
constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

using namespace AsdSip;

// dft stub 从 [sp+0] 裸读 60 字节 tiling, 与 AAPCS64 值传参约定(>16B 按引用, x7=地址)不符,
// GCC 按值直调会错位读栈; 垫片将 tiling 落到 [sp+0..59] 后经 GOT 取弱符号 dft 跳转, x0..x6 透传。
// 垫片仅对 aarch64 (910B 制品) 构建有意义; x86_64 链接 950 制品 (无 dft stub, 判空已拦截, 不可达),
// #else 的 SysV 值传直调兜底仅保链接 (SysV >16B 结构体整体落栈, 无错位)。
#if defined(__aarch64__)
extern "C" void OpsFftCallDft(uint32_t blocks, void* unused, void* stream, void* gm_input, void* gm_dft_matrix,
                              void* gm_output, void* gm_workspace, const OpsFftDftTilingData* tiling);
asm(".text\n"
    ".globl OpsFftCallDft\n"
    "OpsFftCallDft:\n"
    "   adrp x16, :got: dft\n"
    "   ldr  x16, [x16, :got_lo12: dft]\n"
    "   cbz  x16, 9f\n"
    "   sub  sp, sp, #96\n"
    "   stp  x29, x30, [sp, #80]\n"
    "   ldp  x8, x9, [x7]\n        stp  x8, x9, [sp, #0]\n"
    "   ldp  x8, x9, [x7, #16]\n   stp  x8, x9, [sp, #16]\n"
    "   ldp  x8, x9, [x7, #32]\n   stp  x8, x9, [sp, #32]\n"
    "   ldr  x8, [x7, #48]\n       str  x8, [sp, #48]\n"
    "   ldr  w8, [x7, #56]\n       str  w8, [sp, #56]\n"
    "   mov  x7, sp\n"
    "   blr  x16\n"
    "   ldp  x29, x30, [sp, #80]\n"
    "   add  sp, sp, #96\n"
    "   ret\n"
    "9:  ret\n");
#else
extern "C" void OpsFftCallDft(uint32_t blocks, void* unused, void* stream, void* gm_input, void* gm_dft_matrix,
                              void* gm_output, void* gm_workspace, const OpsFftDftTilingData* tiling)
{
    dft(blocks, unused, stream, gm_input, gm_dft_matrix, gm_output, gm_workspace, *tiling);
}
#endif

size_t DFTCore::EstimateWorkspaceSize()
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void DFTCore::Run(Tensor& input, Tensor& output, void* stream, workspace::Workspace& workspace)
{
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();

    // set workspace
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    runInfo.SetScratchDeviceAddr((uint8_t*)workspace.allocate(bufferSize));

    runInfo.SetStream(stream);
    launchParam.GetInTensor(0).data = input.data;
    launchParam.GetOutTensor(0).data = output.data;
    // 输出 desc 全量取 launchParam 输出张量（issue #379）：旧实现整体赋值 input.desc，
    // R2C/C2R 场景下 dtype/dims/strides 被输入（复数）描述污染；dims 为空时先校验
    if (input.desc.dims.empty() || launchParam.GetOutTensor(0).desc.dims.empty()) {
        ASDSIP_LOG(ERROR) << "DFTCore empty tensor dims, run rejected.";
        workspace.recycleLast();
        return;
    }
    output.desc = launchParam.GetOutTensor(0).desc;
    // 下发失败抛出异常，由 exec 入口 try/catch 统一翻译为错误码（issue #568）
    if (!kernel->Run(launchParam, runInfo).Ok()) {
        ASDSIP_LOG(ERROR) << "DFTCore kernel launch failed.";
        throw std::runtime_error("DFTCore kernel launch failed.");
    }

    workspace.recycleLast();

    ASDSIP_LOG(INFO) << "DFTCore run success.";
    return;
}

// stub 判空仅防护绕过统一拦截的直接调用(缺失走 internal)
void DFTCore::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (AsdSip::UseOpsFftKernelBackend() && AsdSip::OpsFftDftStubAvailable()) {
        RunViaOpsFft(input, output, stream, workspace);
        return;
    }
    FftOperation::Run(input, output, stream, workspace);
}

bool DFTCore::OpsFftBackendStubReady() const { return AsdSip::OpsFftDftStubAvailable(); }

// ops-fft 直调: 常量/tiling 复用 plan 期产物 (tiling 值传)
void DFTCore::RunViaOpsFft(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    if (opsTilingHost.size() < sizeof(OpsFftDftTilingData)) {
        ASDSIP_LOG(ERROR) << "ops-fft backend: dft tiling host copy is missing";
        throw std::runtime_error("ops-fft backend: dft tiling host copy is missing");
    }

    // 常量首次 Run 拷贝至 device, plan 级缓存
    if (opsRotationMatrix == nullptr) {
        if (rotationMatrix == nullptr || rotationMatrix->hostData == nullptr || rotationMatrix->dataSize == 0) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: rotationMatrix is empty";
            throw std::runtime_error("ops-fft backend: rotationMatrix is empty");
        }
        if (!OpsFftUploadConst(&opsRotationMatrix, rotationMatrix->hostData, rotationMatrix->dataSize)) {
            ASDSIP_LOG(ERROR) << "ops-fft backend: rotationMatrix upload failed";
            throw std::runtime_error("ops-fft backend: rotationMatrix upload failed");
        }
    }

    // workspace: 与 sip kernel 同源同尺寸 (kernel 自身仅需极小 scratch)
    const KernelInfo& kernelInfo = kernel->GetKernelInfo();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    void* scratch = workspace.allocate(bufferSize);

    // blocks: 全核 launch, plan 级缓存
    if (opsCachedBlocks == 0) {
        uint32_t coreNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
        opsCachedBlocks = AsdSip::OpsFftAllCoreNum(coreNum);
    }

    OpsFftDftTilingData tiling{};
    std::memcpy(&tiling, opsTilingHost.data(), sizeof(tiling));
    // 值传递 tiling 须经裸栈垫片调用(见 OpsFftCallDft 注释), 不能直接按 C 值参调 dft
    OpsFftCallDft(opsCachedBlocks, nullptr, stream, input, opsRotationMatrix, output, scratch, &tiling);

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "DFTCore run via ops-fft kernel (blocks=" << opsCachedBlocks << ") success.";
}

void DFTCore::DestroyInDevice()
{
    // destroy tiling data in device
    uint8_t* deviceLaunchBuffer = nullptr;
    deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
    if (opsRotationMatrix != nullptr) {
        MkiRtMemFreeDevice(opsRotationMatrix);
        opsRotationMatrix = nullptr;
    }
}

AspbStatus DFTCore::InitRotationMatrix()
{
    int64_t fftN = static_cast<int64_t>(problemDesc.nDoing);
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;

    std::function<FFTensor*()> func = [=]() -> FFTensor* {
        FFTensor* rotationMatrixPtr = new FFTensor;
        FFTensor& rotationMatrix_ = *rotationMatrixPtr;

        float* rotationMatrixHost = nullptr;
        try {
            rotationMatrixHost = new float[outSize * inSize];
        } catch (std::bad_alloc& e) {
            delete rotationMatrixPtr;
            ASDSIP_LOG(ERROR) << "rotationMatrixHost malloc failed: ";
            throw std::runtime_error("rotationMatrixHost malloc failed:.");
        }

        std::vector<float> cosTable(static_cast<size_t>(fftN));
        std::vector<float> sinTable(static_cast<size_t>(fftN));
        for (int64_t i = 0; i < fftN; i++) {
            cosTable[i] = cos(K_2PI * i / fftN);
            sinTable[i] = sin(K_2PI * i / fftN);
        }
        for (int64_t i = 0; i < fftN; i++) {
            for (int64_t j = 0; j < fftN; j++) {
                *(rotationMatrixHost + (2 * i) * (2 * fftN) + 2 * j) = cosTable[(i * j) % fftN];
                *(rotationMatrixHost + (2 * i) * (2 * fftN) + 2 * j + 1) = -1 * (sinTable[(i * j) % fftN]);
                *(rotationMatrixHost + (2 * i + 1) * (2 * fftN) + 2 * j) = 1 * (sinTable[(i * j) % fftN]);
                *(rotationMatrixHost + (2 * i + 1) * (2 * fftN) + 2 * j + 1) = cosTable[(i * j) % fftN];
            }
        }

        rotationMatrix_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {outSize, inSize}, {}, 0};
        rotationMatrix_.hostData = rotationMatrixHost;
        rotationMatrix_.dataSize = sizeof(float) * static_cast<size_t>(outSize * inSize);

        return rotationMatrixPtr;
    };

    CoeffKey key = {coreType, 0, {outSize}, 0};
    rotationMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "DFTCore init rotationMatrix success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void DFTCore::InitRadix() {}

bool DFTCore::PreAllocateInDevice()
{
    if (InitRotationMatrix() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }
    ASDSIP_LOG(INFO) << "DFTCore PreAllocateInDevice success.";
    return true;
}

AspbStatus DFTCore::InitTactic()
{
    OpParam::Dft param = {problemDesc.nDoing, problemDesc.batch, 1 - int(problemDesc.forward)};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorIn;
    Tensor tensorOut;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, problemDesc.nDoing}, {}, 0};
    tensorIn.dataSize = problemDesc.batch * problemDesc.nDoing * K_SIZE_OF_COMPLEX64;
    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    // Both Cube backends use the cached forward matrix. Inverse direction is
    // expressed by transB in tiling, without a second host-side transpose.
    launchParam.AddInTensor(*rotationMatrix);
    launchParam.AddOutTensor(tensorOut);
    Operation* op = Ops::Instance().GetOperationByName(std::string("DftOperation"));
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
    // ops-fft 后端 dft stub 的 tiling 为值传递, 保留 host 副本供 RunViaOpsFft 装载
    opsTilingHost.assign(hostLaunchBuffer.begin(), hostLaunchBuffer.end());
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

    ASDSIP_LOG(INFO) << "DFTCore init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
