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
#include <cmath>
#include <complex>
#include <fstream>
#include <random>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
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
