/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <complex>
#include <random>
#include <string>
#include <vector>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr double ISTFT_RTOL = 1e-3;
constexpr double ISTFT_ATOL = 1e-4;

std::string GetIstftOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    return current_dir_env != nullptr ? std::string(current_dir_env) : std::string(".");
}

// Always-positive window (min value 0.2) so the w^2 overlap-add envelope never
// degenerates; must stay in sync with gen_data_istft.py which reads the saved bin.
std::vector<float> MakeIstftWindow(int64_t nFft)
{
    std::vector<float> window(static_cast<size_t>(nFft));
    for (int64_t k = 0; k < nFft; ++k) {
        window[static_cast<size_t>(k)] = static_cast<float>(
            0.6 + 0.4 * std::sin(2.0 * M_PI * static_cast<double>(k) / static_cast<double>(nFft) + 0.7));
    }
    return window;
}

// execTimes: 同一 plan + 同一 window 的连续执行次数（issue #191）。>1 时额外校验
// ① 每次执行输出都与单次参考值（golden）一致 ② 执行后用户 window 张量内容不变。
void RunIstftTest(int64_t batch, int64_t nFft, int64_t hop, int64_t frames, const std::string& suffix,
                  int64_t execTimes = 1)
{
    static std::string destPath = PrepareDataDirOnce(GetIstftOutputDirectory(), "fft/istft", "istft_data");
    std::string dataDir = destPath + "/istft_data";

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    // input spec: (B, nFft, T) complex64; output: (B, hop*(T-1)) complex64
    const int64_t outSignalLen = hop * (frames - 1);
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nFft, frames}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, outSignalLen}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    SaveTensorToBin(inputTensor, dataDir + "/complex64_input_0.bin");

    // window generated on C++ side and saved so that golden shares bit-exact values
    std::vector<float> window = MakeIstftWindow(nFft);
    std::string windowFile = dataDir + "/float_window" + suffix + ".bin";
    FILE* fp = fopen(windowFile.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(window.data(), sizeof(float), window.size(), fp);
    fclose(fp);

    std::string genDataCmd = "cd " + ShellQuote(dataDir) + " && python3 gen_data_istft.py --batch " +
                             std::to_string(batch) + " --nFft " + std::to_string(nFft) + " --hop " +
                             std::to_string(hop) + " --frames " + std::to_string(frames) + " --suffix " + suffix;
    // golden 生成失败必须失败用例，否则残留旧参数的 golden 会造成"假通过"
    int genRes = system(genDataCmd.c_str());
    ASSERT_EQ(genRes, 0);
    std::string goldenFile = dataDir + "/complex64_golden_istft" + suffix + ".bin";

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    aclTensor* aclWindow = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* windowDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * nFft * frames);
    auto* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (size_t i = 0; i < inputHostData.size(); ++i) {
        inputHostData[i] = inputDataPtr[i];
    }
    std::vector<std::complex<float>> outputHostData(batch * outSignalLen, std::complex<float>(0.0f, 0.0f));

    auto ret = CreateAclTensor(inputHostData, {batch, nFft, frames}, &inputDeviceAddr, aclDataType::ACL_COMPLEX64,
                               &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, {batch, outSignalLen}, &outputDeviceAddr, aclDataType::ACL_COMPLEX64,
                          &aclOutput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(window, {nFft}, &windowDeviceAddr, aclDataType::ACL_FLOAT, &aclWindow);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    // center=true, normalized=false, onesided=false, length=0, returnComplex=true
    fftStatus = asdFftIstftMakePlan(handle, aclInput, nFft, hop, nFft, true, false, false, 0, true);
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

    // issue #191：同一 plan + 同一 window 连续执行 execTimes 次；首轮输出快照用于重复执行一致性比对
    const size_t outBytes = static_cast<size_t>(batch * outSignalLen) * sizeof(std::complex<float>);
    std::vector<std::complex<float>> outFirstSnapshot;
    for (int64_t round = 1; round <= execTimes; ++round) {
        fftStatus = asdFftExecIstft(handle, aclInput, aclWindow, aclOutput);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

        fftStatus = asdFftSynchronize(handle);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

        if (round < execTimes) {
            outFirstSnapshot.resize(static_cast<size_t>(batch * outSignalLen));
            ret = aclrtMemcpy(outFirstSnapshot.data(), outBytes, outputDeviceAddr, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            ASSERT_EQ(ret, ::ACL_SUCCESS);
        }
    }

    // issue #191 验收②：执行后用户 window 张量内容不变（const 语义，不被 w^2 回写污染）
    if (execTimes > 1) {
        std::vector<float> windowAfter(window.size(), 0.0f);
        ret = aclrtMemcpy(windowAfter.data(), window.size() * sizeof(float), windowDeviceAddr,
                          window.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        ASSERT_EQ(memcmp(windowAfter.data(), window.data(), window.size() * sizeof(float)), 0);
    }

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData, outputTensor.dataSize, outputDeviceAddr, outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclDestroyTensor(aclWindow);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(windowDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 比对必须在 OpTestEnd 之前完成——OpTestEnd 释放 tensor host 内存
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int64_t numel = batch * outSignalLen;
    int res = CompareGoldenWithOutput(goldenFile, outputTensor.hostData, static_cast<size_t>(numel), true, ISTFT_RTOL,
                                      ISTFT_ATOL, &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ")" << std::endl;
    ASSERT_EQ(res, 0);

    // issue #191 验收①：首轮输出同样与单次执行参考值一致（两次均对 golden 即两两一致）
    if (execTimes > 1) {
        res = CompareGoldenWithOutput(goldenFile, outFirstSnapshot.data(), static_cast<size_t>(numel), true, ISTFT_RTOL,
                                      ISTFT_ATOL, &maxAbsErr, &maxRelErr);
        std::cout << "first-exec compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res
                  << ", max_abs=" << maxAbsErr << ", max_rel=" << maxRelErr << ")" << std::endl;
        ASSERT_EQ(res, 0);
    }

    OpTestEnd(deviceId, context, stream);
}
} // namespace

TEST(TestFftIstft, TestIstftBatch2N64Hop16) { RunIstftTest(2, 64, 16, 8, "_a"); }
TEST(TestFftIstft, TestIstftBatch1N128Hop32) { RunIstftTest(1, 128, 32, 6, "_b"); }
TEST(TestFftIstft, TestIstftBatch3N256Hop64) { RunIstftTest(3, 256, 64, 5, "_c"); }

// issue #191 回归：同一 plan + 同一 window 连续执行两次，且 hop*(frames-1) < nFft
TEST(TestFftIstft, TestIstftRepeatExecWindowUnchanged) { RunIstftTest(1, 128, 16, 4, "_rep", 2); }
