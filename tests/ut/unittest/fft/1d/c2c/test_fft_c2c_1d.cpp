/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <random>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {
std::string GetC2COutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}

void RunFftC2CTest(int64_t batch, int64_t nfft, asdFftDirection direction)
{
    // suite 级一次性目录准备（替代原每用例 3 次 system(mkdir/cp/chmod)，UT 提效）
    static std::string destPath = PrepareDataDirOnce(GetC2COutputDirectory(), "fft/1d/c2c", "c2c_data");
    std::string dataDir = destPath + "/c2c_data";

    std::string dirStr = (direction == asdFftDirection::ASCEND_FFT_FORWARD) ? "forward" : "inverse";
    // golden 按 direction 区分文件名，消除正/逆用例共用文件的时序耦合
    std::string goldenFile = dataDir + "/complex64_golden_c2c" +
                             (direction == asdFftDirection::ASCEND_FFT_FORWARD ? "" : "_inv") + ".bin";

    int deviceId = 0;
    int64_t inSignal = nfft;
    int64_t outSignal = nfft;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, inSignal}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, outSignal}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/c2c_data/" + filename);
    }

    // golden 生成：input bin 由本用例随机生成，golden 必须按本用例参数重算，
    // 目录拷贝已一次化，python 调用保留（np.fft 精度基准）
    std::string gen_data_cmd = "cd " + ShellQuote(dataDir) + " && python3 gen_data.py --batch " +
                               std::to_string(batch) + " --nfft " + std::to_string(nfft) + " --direction " + dirStr;
    // golden 生成失败必须失败用例，否则残留旧参数的 golden 会造成"假通过"
    int genRes = system(gen_data_cmd.c_str());
    ASSERT_EQ(genRes, 0);

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * inSignal);
    std::complex<float>* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < batch * inSignal; i++) {
        inputHostData[i] = inputDataPtr[i];
    }

    std::vector<std::complex<float>> outputHostData(batch * outSignal, std::complex<float>(0.0f, 0.0f));
    std::vector<int64_t> inShape = {batch, inSignal};
    std::vector<int64_t> outShape = {batch, outSignal};

    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2C, direction, batch);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        fftStatus = asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftExecC2C(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    SaveOutTensorToBin(outputTensor, destPath + "/c2c_data/" + outFilename);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 比对 C++ 化：直接用 host 内存与 golden bin 对比（isclose 语义复刻），
    // 消除每用例一次 python3 进程启动开销（UT 提效）。
    // 注意：必须在 OpTestEnd 之前完成——OpTestEnd 释放 tensor host 内存
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int64_t numel = batch * nfft;
    int res = CompareGoldenWithOutput(goldenFile, outputTensor.hostData, static_cast<size_t>(numel), true, 1e-3, 1e-5,
                                      &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ")" << std::endl;
    ASSERT_EQ(res, 0);

    OpTestEnd(deviceId, context, stream);
}

} // namespace

// Pure radix forward tests (<= 256, DFT path)
TEST(TestFftC2C1d, TestC2CForwardRadix2Dft) { RunFftC2CTest(1, 64, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix3Dft) { RunFftC2CTest(1, 81, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix5Dft) { RunFftC2CTest(1, 125, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix7Dft) { RunFftC2CTest(1, 49, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix11) { RunFftC2CTest(1, 121, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix13) { RunFftC2CTest(1, 169, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix17Dft) { RunFftC2CTest(1, 17, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix19Dft) { RunFftC2CTest(1, 19, asdFftDirection::ASCEND_FFT_FORWARD); }

// Pure radix forward tests (> 256, arch35 path)
TEST(TestFftC2C1d, TestC2CForwardRadix2) { RunFftC2CTest(1, 1024, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix3) { RunFftC2CTest(1, 729, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix5) { RunFftC2CTest(1, 625, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix7) { RunFftC2CTest(1, 343, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix17) { RunFftC2CTest(1, 289, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix19) { RunFftC2CTest(1, 361, asdFftDirection::ASCEND_FFT_FORWARD); }

// Large signal tests (forward, up to 32768)
TEST(TestFftC2C1d, TestC2CForwardRadix2Large) { RunFftC2CTest(1, 4096, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix3Large) { RunFftC2CTest(1, 6561, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardRadix2Max) { RunFftC2CTest(1, 32768, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardMixedLarge) { RunFftC2CTest(1, 4620, asdFftDirection::ASCEND_FFT_FORWARD); }

// Mixed radix tests (forward)
TEST(TestFftC2C1d, TestC2CForwardMixed2357) { RunFftC2CTest(1, 210, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardMixed235711) { RunFftC2CTest(1, 2310, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardMixed1113) { RunFftC2CTest(1, 143, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardMixed1719) { RunFftC2CTest(1, 323, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardMixed360) { RunFftC2CTest(1, 360, asdFftDirection::ASCEND_FFT_FORWARD); }

// Batch tests (forward)
TEST(TestFftC2C1d, TestC2CForwardBatch2) { RunFftC2CTest(2, 1024, asdFftDirection::ASCEND_FFT_FORWARD); }
TEST(TestFftC2C1d, TestC2CForwardBatch2Mixed) { RunFftC2CTest(2, 210, asdFftDirection::ASCEND_FFT_FORWARD); }

// Pure radix inverse tests (<= 256, DFT path)
TEST(TestFftC2C1d, TestC2CInverseRadix2Dft) { RunFftC2CTest(1, 64, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix3Dft) { RunFftC2CTest(1, 81, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix5Dft) { RunFftC2CTest(1, 125, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix7Dft) { RunFftC2CTest(1, 49, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix11) { RunFftC2CTest(1, 121, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix13) { RunFftC2CTest(1, 169, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix17Dft) { RunFftC2CTest(1, 17, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix19Dft) { RunFftC2CTest(1, 19, asdFftDirection::ASCEND_FFT_INVERSE); }

// Pure radix inverse tests (> 256, arch35 path)
TEST(TestFftC2C1d, TestC2CInverseRadix2) { RunFftC2CTest(1, 1024, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix3) { RunFftC2CTest(1, 729, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix5) { RunFftC2CTest(1, 625, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix7) { RunFftC2CTest(1, 343, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix17) { RunFftC2CTest(1, 289, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix19) { RunFftC2CTest(1, 361, asdFftDirection::ASCEND_FFT_INVERSE); }

// Large signal inverse tests (up to 32768)
TEST(TestFftC2C1d, TestC2CInverseRadix2Large) { RunFftC2CTest(1, 4096, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseRadix2Max) { RunFftC2CTest(1, 32768, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseMixedLarge) { RunFftC2CTest(1, 4620, asdFftDirection::ASCEND_FFT_INVERSE); }

// Mixed radix inverse tests
TEST(TestFftC2C1d, TestC2CInverseMixed2357) { RunFftC2CTest(1, 210, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseMixed235711) { RunFftC2CTest(1, 2310, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2C1d, TestC2CInverseBatch2) { RunFftC2CTest(2, 1024, asdFftDirection::ASCEND_FFT_INVERSE); }

// ---- 负向校验：不依赖 golden 数据，仅断言返回码（issue #124/#137） ----

// 2D plan 不支持 C2C_SEP：应显式返回参数错误而非生成 0 步 plan（issue #124）
TEST(TestFftPlanNegative, MakePlan2DRejectC2CSep)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftMakePlan2D(handle, 64, 64, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 同一 handle 重复初始化应被拒绝，而非 steps 累积追加（issue #137）
TEST(TestFftPlanNegative, RepeatedInitRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftMakePlan1D(handle, 128, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan2D(handle, 64, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 2D 派生 batch 乘积（batchSize * fftSizeX）超过 INT32_MAX 时应返回参数错误，
// 而非截断为负/错值静默产出错误 plan（issue #214）
TEST(TestFftPlanNegative, MakePlan2DDerivedBatchOverflowRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    // batch(2^26) * fftSizeX(64) = 2^32 > INT32_MAX，batchSize 本身未超 1D 的 #175 上限；
    // fftSizeY=256 超出 2D 融合核心上限(128)，确保走 addFFTSteps 的派生 batch 乘积路径
    EXPECT_EQ(
        asdFftMakePlan2D(handle, 64, 256, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 67108864),
        AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 跨变体误用：C2R plan 经 C2C/R2C 入口、C2C plan 经 C2R 入口均应返回参数错误，
// 而非通过 match 检查后以错误核心静默执行（issue #213）
TEST(TestFftPlanNegative, CrossVariantExecRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    // 与 C2C 入口 dtype/形状要求完全匹配的 complex64 张量：修复前可越过全部 match 检查
    std::vector<std::complex<float>> inputHostData(16, std::complex<float>(1.0f, 0.0f));
    std::vector<std::complex<float>> outputHostData(16, std::complex<float>(0.0f, 0.0f));
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    ASSERT_EQ(CreateAclTensor(inputHostData, {1, 16}, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput), 0);
    ASSERT_EQ(CreateAclTensor(outputHostData, {1, 16}, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput), 0);

    asdFftHandle c2rHandle;
    ASSERT_EQ(asdFftCreate(c2rHandle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(c2rHandle, 16, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftExecC2C(c2rHandle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftExecR2C(c2rHandle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(c2rHandle);

    asdFftHandle c2cHandle;
    ASSERT_EQ(asdFftCreate(c2cHandle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(c2cHandle, 16, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftExecC2R(c2cHandle, aclInput, aclOutput), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(c2cHandle);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// ---- 负向校验：MakePlan1D fftType 白名单（issue #551） ----

// STFT_* 与非法枚举值 1D 不支持：应返回参数错误，且 plan 不得被标记为已初始化——
// 同句柄改用合法类型重新初始化必须成功（修复前 STFT 调用返回成功并占用句柄，
// 后续 exec 全被拒绝、句柄永久不可执行）
TEST(TestFftPlanNegative, MakePlan1DRejectStftType)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_STFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_STFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_STFT_R2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, static_cast<asdFftType>(0x99), asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // 拒绝路径不得改写 plan 状态：合法类型在同一句柄上重新初始化应成功
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// ---- 负向校验：MakePlan1D/2D/3D direction 与 1D dimType 白名单（issue #562） ----

// direction 非法枚举值（未初始化结构体的 0、越界值）：isForward() 以 direction ==
// ASCEND_FFT_FORWARD 判定正反向，非法值一律静默按反向变换执行并返回成功——应显式
// 拒绝且 plan 不得被标记为已初始化，同句柄合法参数重新初始化应成功
TEST(TestFftPlanNegative, MakePlanRejectInvalidDirection)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, static_cast<asdFftDirection>(0), 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, static_cast<asdFftDirection>(0x99), 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan2D(handle, 8, 8, asdFftType::ASCEND_FFT_C2C, static_cast<asdFftDirection>(0), 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan3D(handle, 8, 8, 8, asdFftType::ASCEND_FFT_C2C, static_cast<asdFftDirection>(0), 1),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // 拒绝路径不得改写 plan 状态：合法参数在同一句柄上重新初始化应成功
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 1D dimType 非法枚举值：落入 else 分支会静默按 HORIZONTAL 处理并绕过 950 平台
// VERTICAL+batchSize>1 的不支持拦截——应显式拒绝，plan 未被标记初始化
TEST(TestFftPlanNegative, MakePlan1DRejectInvalidDimType)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1,
                               static_cast<asdFft1dDimType>(0)),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1,
                               static_cast<asdFft1dDimType>(0x99)),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    // 拒绝路径不得改写 plan 状态：合法参数在同一句柄上重新初始化应成功
    EXPECT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1),
              AsdSip::ErrorType::ACL_SUCCESS);
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// ---- 负向校验：asdFftExecC2CSeparated plan 形状一致性（issue #550） ----

namespace {
// 形状不符用例共用：plan 为 batch=8、fftSize=64 的 C2C_SEP，传入 FLOAT 张量仅形状
// 与 plan 不符（dtype 与四张量内部一致性均满足），应返回参数错误且不直达核心
void RunSepShapeMismatchCase(const std::vector<int64_t>& shape)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    std::vector<float> hostData(numel, 1.0f);
    void* deviceAddr = nullptr;
    aclTensor* aclTensorPtr = nullptr;
    ASSERT_EQ(CreateAclTensor(hostData, shape, &deviceAddr, aclDataType::ACL_FLOAT, &aclTensorPtr), 0);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD, 8),
              AsdSip::ErrorType::ACL_SUCCESS);
    // 四张量传同一指针：内部一致性 trivially 满足，仅形状 vs plan 校验生效
    EXPECT_EQ(asdFftExecC2CSeparated(handle, aclTensorPtr, aclTensorPtr, aclTensorPtr, aclTensorPtr),
              AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    asdFftDestroy(handle);

    aclDestroyTensor(aclTensorPtr);
    aclrtFree(deviceAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
} // namespace

// fftSize 不符：(8, 32) 尾维 32 != plan.fftSize 64
TEST(TestFftPlanNegative, ExecC2CSeparatedFftSizeMismatchRejected) { RunSepShapeMismatchCase({8, 32}); }

// batch 维错位：(4, 64) 前导元素规模 4 != plan.batch 8
TEST(TestFftPlanNegative, ExecC2CSeparatedBatchMismatchRejected) { RunSepShapeMismatchCase({4, 64}); }

// 缺失 batch 维：(64,) 单维张量，前导元素规模 1 != plan.batch 8
TEST(TestFftPlanNegative, ExecC2CSeparatedMissingBatchDimRejected) { RunSepShapeMismatchCase({64}); }

// 3D plan 形状不符：plan 为 batch=2、(8,8,8) 的 C2C_SEP，(2,8,8,4) 尾三维 fftSize 不符应拒绝；
// (8,8,8) 尾三维与 plan.fftSizes 全部匹配，仅因缺失 batch 维（前导元素规模 1 != plan.batch 2）
// 被拒绝——与尾维匹配逻辑无关
TEST(TestFftPlanNegative, ExecC2CSeparatedPlan3DShapeMismatchRejected)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan3D(handle, 8, 8, 8, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD, 2),
              AsdSip::ErrorType::ACL_SUCCESS);

    for (const auto& shape : std::vector<std::vector<int64_t>>{{2, 8, 8, 4}, {8, 8, 8}}) {
        int64_t numel = 1;
        for (auto d : shape) {
            numel *= d;
        }
        std::vector<float> hostData(numel, 1.0f);
        void* deviceAddr = nullptr;
        aclTensor* aclTensorPtr = nullptr;
        ASSERT_EQ(CreateAclTensor(hostData, shape, &deviceAddr, aclDataType::ACL_FLOAT, &aclTensorPtr), 0);
        EXPECT_EQ(asdFftExecC2CSeparated(handle, aclTensorPtr, aclTensorPtr, aclTensorPtr, aclTensorPtr),
                  AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        aclDestroyTensor(aclTensorPtr);
        aclrtFree(deviceAddr);
    }
    asdFftDestroy(handle);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

namespace {
// 分离执行合法形状正向链路共用：创建 4 个同形状 FLOAT 张量（输入填 1、输出清零）、
// 建 plan（由调用方给出 1D/3D 差异）、配 workspace/stream、执行并同步，全链断言成功
template <typename MakePlan>
void RunSeparatedLegalExec(const std::vector<int64_t>& shape, MakePlan makePlan, MkiRtStream stream)
{
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    std::vector<float> hostData(numel, 1.0f);
    std::vector<float> outputHostData(numel, 0.0f);
    void* deviceAddrs[4] = {nullptr, nullptr, nullptr, nullptr};
    aclTensor* tensors[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 4; i++) {
        const std::vector<float>& src = i < 2 ? hostData : outputHostData;
        ASSERT_EQ(CreateAclTensor(src, shape, &deviceAddrs[i], aclDataType::ACL_FLOAT, &tensors[i]), 0);
    }

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(makePlan(handle), AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    ASSERT_EQ(asdFftGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
        ASSERT_EQ(asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    ASSERT_EQ(asdFftSetStream(handle, static_cast<aclrtStream>(stream)), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdFftExecC2CSeparated(handle, tensors[0], tensors[1], tensors[2], tensors[3]),
              AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftSynchronize(handle), AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);
    for (int i = 0; i < 4; i++) {
        aclDestroyTensor(tensors[i]);
        aclrtFree(deviceAddrs[i]);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
}
} // namespace

// 合法形状回归：与 plan 完全一致的 (batch, fftSize) 分离张量应正常执行并同步成功
// （守卫不得误拒文档约定形状）
TEST(TestFftPlanNegative, ExecC2CSeparatedLegalShapeExecutes)
{
    // 1D C2C_SEP 仅 910B 路由到分离核心（getCore 910B 分支）；950 落 FFTCoreAny
    // （SupportsSeparatedExec=false）执行期必返回 INTERNAL_ERROR，正向断言仅对 910B 有效
    if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_910B) {
        GTEST_SKIP() << "1D C2C_SEP separated cores are routed only on Ascend910B.";
    }
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    auto makePlan = [](asdFftHandle handle) {
        return asdFftMakePlan1D(handle, 64, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD, 8);
    };
    RunSeparatedLegalExec({8, 64}, makePlan, stream);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 3D plan 合法形状回归：DddCoreSep 冻结的启动张量形状即 (batch, fftX, fftY, fftZ)，合法 4D
// 张量应全链执行成功（守卫不得误拒文档约定形状，尤其不得把改写后的多维 fftStrides 误当
// batch 推导依据）
TEST(TestFftPlanNegative, ExecC2CSeparatedPlan3DLegalShapeExecutes)
{
    // 3D C2C_SEP 仅 910B 存在分离核心（GetCore3D 在 950 返回 nullptr，plan 创建即失败）
    if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_910B) {
        GTEST_SKIP() << "3D C2C_SEP separated cores are routed only on Ascend910B.";
    }
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    auto makePlan = [](asdFftHandle handle) {
        return asdFftMakePlan3D(handle, 8, 8, 8, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD,
                                2);
    };
    RunSeparatedLegalExec({2, 8, 8, 8}, makePlan, stream);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// ---- 负向校验：asdFftExecC2R/asdFftExecR2C 缺 batch 张量（issue #561） ----

namespace {
// C2R/R2C 缺 batch 用例共用：plan 为 batch=8、fftSize=64，张量 dtype 与尾维均与 plan
// 匹配（C2R 侧尾维按 N/2+1 换算），仅一侧张量前导元素规模与 plan.batchSize 不符——
// 修复前可越过 match 直达核心按 plan.batchSize 读写造成设备越界，应返回参数错误
void RunRealExecBatchMismatchCase(asdFftType planType, const std::vector<int64_t>& inShape,
                                  const std::vector<int64_t>& outShape)
{
    auto shapeNumel = [](const std::vector<int64_t>& shape) {
        int64_t numel = 1;
        for (auto d : shape) {
            numel *= d;
        }
        return numel;
    };
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    void* inAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* aclIn = nullptr;
    aclTensor* aclOut = nullptr;
    if (planType == asdFftType::ASCEND_FFT_C2R) {
        std::vector<std::complex<float>> inHost(shapeNumel(inShape), std::complex<float>(1.0f, 0.0f));
        std::vector<float> outHost(shapeNumel(outShape), 0.0f);
        ASSERT_EQ(CreateAclTensor(inHost, inShape, &inAddr, aclDataType::ACL_COMPLEX64, &aclIn), 0);
        ASSERT_EQ(CreateAclTensor(outHost, outShape, &outAddr, aclDataType::ACL_FLOAT, &aclOut), 0);
    } else {
        std::vector<float> inHost(shapeNumel(inShape), 1.0f);
        std::vector<std::complex<float>> outHost(shapeNumel(outShape), std::complex<float>(0.0f, 0.0f));
        ASSERT_EQ(CreateAclTensor(inHost, inShape, &inAddr, aclDataType::ACL_FLOAT, &aclIn), 0);
        ASSERT_EQ(CreateAclTensor(outHost, outShape, &outAddr, aclDataType::ACL_COMPLEX64, &aclOut), 0);
    }

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 64, planType, asdFftDirection::ASCEND_FFT_FORWARD, 8),
              AsdSip::ErrorType::ACL_SUCCESS);
    if (planType == asdFftType::ASCEND_FFT_C2R) {
        EXPECT_EQ(asdFftExecC2R(handle, aclIn, aclOut), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    } else {
        EXPECT_EQ(asdFftExecR2C(handle, aclIn, aclOut), AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    }
    asdFftDestroy(handle);

    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(inAddr);
    aclrtFree(outAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
} // namespace

// C2R 入口输入侧缺 batch：(1, 33) 尾维 33 == 64/2+1 但前导元素规模 1 != plan.batch 8
TEST(TestFftPlanNegative, ExecC2RInputBatchMismatchRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_C2R, {1, 33}, {8, 64});
}

// C2R 入口输出侧缺 batch：输入 (8, 33) 合法，输出 (1, 64) 尾维 64 == fftSize 但前导元素
// 规模 1 != plan.batch 8（输出 matcher 为 matchR2C_）
TEST(TestFftPlanNegative, ExecC2ROutputBatchMismatchRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_C2R, {8, 33}, {1, 64});
}

// R2C 入口输入侧缺 batch：(1, 64) 尾维 64 == fftSize 但前导元素规模 1 != plan.batch 8
TEST(TestFftPlanNegative, ExecR2CInputBatchMismatchRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_R2C, {1, 64}, {8, 33});
}

// R2C 入口输出侧缺 batch：输入 (8, 64) 合法，输出 (1, 33) 尾维 33 == 64/2+1 但前导元素
// 规模 1 != plan.batch 8（输出 matcher 为 matchC2R_）
TEST(TestFftPlanNegative, ExecR2COutputBatchMismatchRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_R2C, {8, 64}, {1, 33});
}

// 缺失 batch 维（单维张量）：(33,) 尾维与 C2R 输入匹配但无 batch 维，前导元素规模
// 1 != plan.batch 8；同理 (64,) 对 R2C 输入
TEST(TestFftPlanNegative, ExecC2RInputMissingBatchDimRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_C2R, {33}, {8, 64});
}

TEST(TestFftPlanNegative, ExecR2CInputMissingBatchDimRejected)
{
    RunRealExecBatchMismatchCase(asdFftType::ASCEND_FFT_R2C, {64}, {8, 33});
}

namespace {
void RunC2CShapeMismatchCase(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                             asdFft1dDimType dimType, int64_t batchSize, AsdSip::ErrorType expectedExec)
{
    auto shapeNumel = [](const std::vector<int64_t>& shape) {
        int64_t numel = 1;
        for (auto d : shape) {
            numel *= d;
        }
        return numel;
    };
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    void* inAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* aclIn = nullptr;
    aclTensor* aclOut = nullptr;
    std::vector<std::complex<float>> inHost(shapeNumel(inShape), std::complex<float>(1.0f, 0.0f));
    std::vector<std::complex<float>> outHost(shapeNumel(outShape), std::complex<float>(0.0f, 0.0f));
    ASSERT_EQ(CreateAclTensor(inHost, inShape, &inAddr, aclDataType::ACL_COMPLEX64, &aclIn), 0);
    ASSERT_EQ(CreateAclTensor(outHost, outShape, &outAddr, aclDataType::ACL_COMPLEX64, &aclOut), 0);

    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftMakePlan1D(handle, 256, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, batchSize,
                               dimType),
              AsdSip::ErrorType::ACL_SUCCESS);
    EXPECT_EQ(asdFftExecC2C(handle, aclIn, aclOut), expectedExec);
    asdFftDestroy(handle);

    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(inAddr);
    aclrtFree(outAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}

// 横向 plan 接受错误 batch：(1, 256) 尾维 256 == fftSize 但首维 1 != plan.batchSize 128（issue #471）
TEST(TestFftPlanNegative, ExecC2CHorizontalBatchMismatchRejected)
{
    RunC2CShapeMismatchCase({1, 256}, {128, 256}, asdFft1dDimType::ASCEND_FFT_HORIZONTAL, 128,
                            AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

// 横向 plan 接受额外 leading 维：rank 不符的张量被静默放行（issue #471）
TEST(TestFftPlanNegative, ExecC2CHorizontalExtraRankRejected)
{
    RunC2CShapeMismatchCase({7, 128, 256}, {7, 128, 256}, asdFft1dDimType::ASCEND_FFT_HORIZONTAL, 128,
                            AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

// 纵向 plan 收到横向布局 (batchSize, fftSize)：修复前尾维 == fftSize 通过 match，
// 内核按 stride 布局处理得到错误结果；修复后按纵向布局契约拒绝
TEST(TestFftPlanNegative, ExecC2CVerticalSwappedShapeRejected)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        GTEST_SKIP() << "vertical 1D FFT with batchSize > 1 is unsupported on Ascend950.";
    }
    RunC2CShapeMismatchCase({128, 256}, {128, 256}, asdFft1dDimType::ASCEND_FFT_VERTICAL, 128,
                            AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
}

// 纵向 stride>1 plan 的合法 (fftSize, batchSize) 张量：match 按纵向契约接受（#472 拦截侧修复），
// 但纵向 stride 核心数值缺陷修复前在 API 边界显式拒绝执行（API_NOT_SUPPORT，
// 与 Setup1dBatchLayout 对 950 平台的拦截口径一致）；950 平台建 plan 即被拦，同上跳过
TEST(TestFftPlanNegative, ExecC2CVerticalStrideNotSupported)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        GTEST_SKIP() << "vertical 1D FFT with batchSize > 1 is unsupported on Ascend950.";
    }
    RunC2CShapeMismatchCase({256, 128}, {256, 128}, asdFft1dDimType::ASCEND_FFT_VERTICAL, 128,
                            AsdSip::ErrorType::ACL_ERROR_API_NOT_SUPPORT);
}

// 单次 1D C2C batch==1 执行并回读输出：batch==1 时纵向退化为 stride==1 plan，
// 与横向内部描述一致，用于纵向/横向两种布局的形状等价比对
void RunC2CBatch1Exec(MkiRtStream stream, asdFft1dDimType dimType, const std::vector<int64_t>& shape,
                      std::vector<std::complex<float>>& outHost)
{
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    std::vector<std::complex<float>> inHost(numel, std::complex<float>(1.0f, 0.5f));
    outHost.assign(numel, std::complex<float>(0.0f, 0.0f));

    void* inAddr = nullptr;
    void* outAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* aclIn = nullptr;
    aclTensor* aclOut = nullptr;
    size_t workSize = 0;

    ASSERT_EQ(CreateAclTensor(inHost, shape, &inAddr, aclDataType::ACL_COMPLEX64, &aclIn), 0);
    ASSERT_EQ(CreateAclTensor(outHost, shape, &outAddr, aclDataType::ACL_COMPLEX64, &aclOut), 0);
    asdFftHandle handle;
    ASSERT_EQ(asdFftCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(
        asdFftMakePlan1D(handle, 256, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, 1, dimType),
        AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftGetWorkspaceSize(handle, workSize), AsdSip::ErrorType::ACL_SUCCESS);
    if (workSize > 0) {
        ASSERT_EQ(aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST),
                  ::ACL_SUCCESS);
        ASSERT_EQ(asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr), AsdSip::ErrorType::ACL_SUCCESS);
    }
    ASSERT_EQ(asdFftSetStream(handle, static_cast<aclrtStream>(stream)), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftExecC2C(handle, aclIn, aclOut), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdFftSynchronize(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(outHost.data(), outHost.size() * sizeof(std::complex<float>), outAddr,
                          outHost.size() * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST),
              ::ACL_SUCCESS);
    asdFftDestroy(handle);
    aclDestroyTensor(aclIn);
    aclDestroyTensor(aclOut);
    aclrtFree(inAddr);
    aclrtFree(outAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }
}

// 纵向 batchSize==1 退化为 stride==1 plan：文档纵向形状 (fftSize, batchSize)=(256, 1)
// 与横向退化形状 (1, 256) 数值等价，两种形状均应被接受且执行结果一致
// （检视意见：batchSize==1 时无法从 plan 字段反推布局，(256, 1) 曾被误拒）
TEST(TestFftPlanNegative, ExecC2CVerticalBatch1SwappedShapeExecutes)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    std::vector<std::complex<float>> outHorizontal;
    RunC2CBatch1Exec(stream, asdFft1dDimType::ASCEND_FFT_HORIZONTAL, {1, 256}, outHorizontal);
    std::vector<std::complex<float>> outVerticalSwapped;
    RunC2CBatch1Exec(stream, asdFft1dDimType::ASCEND_FFT_VERTICAL, {256, 1}, outVerticalSwapped);
    std::vector<std::complex<float>> outVerticalPlain;
    RunC2CBatch1Exec(stream, asdFft1dDimType::ASCEND_FFT_VERTICAL, {1, 256}, outVerticalPlain);

    ASSERT_EQ(outHorizontal.size(), outVerticalSwapped.size());
    for (size_t i = 0; i < outHorizontal.size(); i++) {
        std::complex<float> diff = outHorizontal[i] - outVerticalSwapped[i];
        double relErr = std::abs(diff) / (std::abs(outHorizontal[i]) + 1e-10);
        ASSERT_LT(relErr, 1e-5) << "index " << i << " relErr " << relErr;
    }
    ASSERT_EQ(outHorizontal.size(), outVerticalPlain.size());
    for (size_t i = 0; i < outHorizontal.size(); i++) {
        std::complex<float> diff = outHorizontal[i] - outVerticalPlain[i];
        double relErr = std::abs(diff) / (std::abs(outHorizontal[i]) + 1e-10);
        ASSERT_LT(relErr, 1e-5) << "index " << i << " relErr " << relErr;
    }

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
} // namespace
