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

std::string GetC2ROutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}

void RunFftC2RTest(int64_t batch, int64_t nfft, asdFftDirection direction = asdFftDirection::ASCEND_FFT_FORWARD)
{
    // suite 级一次性目录准备（替代原每用例 3 次 system，UT 提效）
    static std::string destPath = PrepareDataDirOnce(GetC2ROutputDirectory(), "fft/1d/c2r", "c2r_data");
    std::string dataDir = destPath + "/c2r_data";

    std::string dirStr = (direction == asdFftDirection::ASCEND_FFT_FORWARD) ? "forward" : "inverse";
    std::string goldenFile = dataDir + "/float_golden_c2r" +
                             (direction == asdFftDirection::ASCEND_FFT_FORWARD ? "" : "_inv") + ".bin";

    int deviceId = 0;
    int64_t inSignal = nfft / 2 + 1;
    int64_t outSignal = nfft;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, inSignal}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, outSignal}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/c2r_data/" + filename);
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/c2r_data/") + " && python3 gen_data.py --batch " +
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

    std::vector<float> outputHostData(batch * outSignal, 0.0f);
    std::vector<int64_t> inShape = {batch, inSignal};
    std::vector<int64_t> outShape = {batch, outSignal};

    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2R, direction, batch);
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

    fftStatus = asdFftExecC2R(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    SaveOutTensorToBin(outputTensor, destPath + "/c2r_data/" + outFilename);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 比对 C++ 化：golden 为 float32（c2r 实数输出），直接 host 内存比对（UT 提效）。
    // 注意：必须在 OpTestEnd 之前完成——OpTestEnd 释放 tensor host 内存
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int64_t numel = batch * nfft;
    int res = CompareGoldenWithOutput(goldenFile, outputTensor.hostData, static_cast<size_t>(numel), false, 1e-3, 1e-5,
                                      &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ")" << std::endl;
    ASSERT_EQ(res, 0);

    OpTestEnd(deviceId, context, stream);
}

} // namespace

// Pure radix tests (forward)
TEST(TestFftC2R1d, TestC2RForwardRadix2) { RunFftC2RTest(1, 16); }
TEST(TestFftC2R1d, TestC2RForwardRadix2B2) { RunFftC2RTest(2, 32); }
TEST(TestFftC2R1d, TestC2RForwardRadix2B1) { RunFftC2RTest(1, 64); }
TEST(TestFftC2R1d, TestC2RForwardRadix3) { RunFftC2RTest(1, 27); }
TEST(TestFftC2R1d, TestC2RForwardRadix5) { RunFftC2RTest(1, 25); }
TEST(TestFftC2R1d, TestC2RForwardRadix7) { RunFftC2RTest(1, 49); }
TEST(TestFftC2R1d, TestC2RForwardRadix11) { RunFftC2RTest(1, 121); }
// TEST(TestFftC2R1d, TestC2RForwardRadix13) { RunFftC2RTest(1, 169); }
TEST(TestFftC2R1d, TestC2RForwardRadix17) { RunFftC2RTest(1, 289); }
TEST(TestFftC2R1d, TestC2RForwardRadix19) { RunFftC2RTest(1, 361); }

// Mixed radix tests (forward)
TEST(TestFftC2R1d, TestC2RForwardMixed235) { RunFftC2RTest(1, 30); }
TEST(TestFftC2R1d, TestC2RForwardMixed2357) { RunFftC2RTest(1, 210); }
TEST(TestFftC2R1d, TestC2RForwardMixedBatch2) { RunFftC2RTest(2, 30); }

// Large signal tests (forward, > 1024, arch35 path, up to 32768)
TEST(TestFftC2R1d, TestC2RForwardRadix2Large) { RunFftC2RTest(1, 2048); }
TEST(TestFftC2R1d, TestC2RForwardRadix3Large) { RunFftC2RTest(1, 2187); }
TEST(TestFftC2R1d, TestC2RForwardRadix5Large) { RunFftC2RTest(1, 3125); }
// TEST(TestFftC2R1d, TestC2RForwardRadix7Large)   { RunFftC2RTest(1, 2401); }
TEST(TestFftC2R1d, TestC2RForwardMixedLarge) { RunFftC2RTest(1, 2160); }
// TEST(TestFftC2R1d, TestC2RForwardRadix2Max)     { RunFftC2RTest(1, 32768); }
TEST(TestFftC2R1d, TestC2RForwardMixedMax) { RunFftC2RTest(1, 2520); }

// Inverse tests
TEST(TestFftC2R1d, TestC2RInverseRadix2) { RunFftC2RTest(1, 16, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix3) { RunFftC2RTest(1, 27, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix5) { RunFftC2RTest(1, 25, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix7) { RunFftC2RTest(1, 49, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix11) { RunFftC2RTest(1, 121, asdFftDirection::ASCEND_FFT_INVERSE); }
// TEST(TestFftC2R1d, TestC2RInverseRadix13)   { RunFftC2RTest(1, 169, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix17) { RunFftC2RTest(1, 289, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix19) { RunFftC2RTest(1, 361, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseMixed235) { RunFftC2RTest(1, 30, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseMixed2357) { RunFftC2RTest(1, 210, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseBatch2) { RunFftC2RTest(2, 16, asdFftDirection::ASCEND_FFT_INVERSE); }

// Large signal inverse tests (> 1024, arch35 path, up to 32768)
TEST(TestFftC2R1d, TestC2RInverseRadix2Large) { RunFftC2RTest(1, 2048, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix3Large) { RunFftC2RTest(1, 2187, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseRadix5Large) { RunFftC2RTest(1, 3125, asdFftDirection::ASCEND_FFT_INVERSE); }
// TEST(TestFftC2R1d, TestC2RInverseRadix7Large)   { RunFftC2RTest(1, 2401, asdFftDirection::ASCEND_FFT_INVERSE); }
TEST(TestFftC2R1d, TestC2RInverseMixedLarge) { RunFftC2RTest(1, 2160, asdFftDirection::ASCEND_FFT_INVERSE); }
// TEST(TestFftC2R1d, TestC2RInverseRadix2Max)     { RunFftC2RTest(1, 32768, asdFftDirection::ASCEND_FFT_INVERSE); }
