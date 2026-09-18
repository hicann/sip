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
#include <complex>
#include <random>
#include <string>
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
constexpr double C2C3D_RTOL = 1e-3;
constexpr double C2C3D_ATOL = 1e-4;

std::string GetC2C3DOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    return current_dir_env != nullptr ? std::string(current_dir_env) : std::string(".");
}

void RunFftC2C3DTest(int64_t sizeX, int64_t sizeY, int64_t sizeZ, int64_t batch, asdFftDirection direction,
                     const std::string& suffix)
{
    static std::string destPath = PrepareDataDirOnce(GetC2C3DOutputDirectory(), "fft/3d/c2c", "c2c_3d_data");
    std::string dataDir = destPath + "/c2c_3d_data";

    std::string dirStr = (direction == asdFftDirection::ASCEND_FFT_FORWARD) ? "forward" : "inverse";

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, sizeX, sizeY, sizeZ}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, sizeX, sizeY, sizeZ}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    SaveTensorToBin(inputTensor, dataDir + "/complex64_input_0.bin");

    std::string genDataCmd = "cd " + ShellQuote(dataDir) + " && python3 gen_data_3d.py --sizeX " +
                             std::to_string(sizeX) + " --sizeY " + std::to_string(sizeY) + " --sizeZ " +
                             std::to_string(sizeZ) + " --batch " + std::to_string(batch) + " --direction " + dirStr +
                             " --suffix " + suffix;
    // golden 生成失败必须失败用例，否则残留旧参数的 golden 会造成"假通过"
    int genRes = system(genDataCmd.c_str());
    ASSERT_EQ(genRes, 0);
    std::string goldenFile = dataDir + "/complex64_golden_c2c_3d" + suffix + ".bin";

    aclTensor* aclInput = nullptr;
    aclTensor* aclOutput = nullptr;
    void* inputDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * sizeX * sizeY * sizeZ);
    auto* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (size_t i = 0; i < inputHostData.size(); ++i) {
        inputHostData[i] = inputDataPtr[i];
    }
    std::vector<std::complex<float>> outputHostData(batch * sizeX * sizeY * sizeZ, std::complex<float>(0.0f, 0.0f));
    std::vector<int64_t> shape = {batch, sizeX, sizeY, sizeZ};

    auto ret = CreateAclTensor(inputHostData, shape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, shape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan3D(handle, sizeX, sizeY, sizeZ, asdFftType::ASCEND_FFT_C2C, direction,
                                 static_cast<int32_t>(batch));
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

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 比对必须在 OpTestEnd 之前完成——OpTestEnd 释放 tensor host 内存
    double maxAbsErr = 0.0, maxRelErr = 0.0;
    int64_t numel = batch * sizeX * sizeY * sizeZ;
    int res = CompareGoldenWithOutput(goldenFile, outputTensor.hostData, static_cast<size_t>(numel), true, C2C3D_RTOL,
                                      C2C3D_ATOL, &maxAbsErr, &maxRelErr);
    std::cout << "compare result = " << (res == 0 ? 0 : 1) << " (mismatch=" << res << ", max_abs=" << maxAbsErr
              << ", max_rel=" << maxRelErr << ")" << std::endl;
    ASSERT_EQ(res, 0);

    OpTestEnd(deviceId, context, stream);
}
} // namespace

TEST(TestFftC2C3d, TestC2C3DForwardCube8) { RunFftC2C3DTest(8, 8, 8, 1, asdFftDirection::ASCEND_FFT_FORWARD, "_b1"); }
TEST(TestFftC2C3d, TestC2C3DInverseCube16)
{
    RunFftC2C3DTest(16, 16, 16, 2, asdFftDirection::ASCEND_FFT_INVERSE, "_inv");
}
TEST(TestFftC2C3d, TestC2C3DForwardMixedRadix)
{
    RunFftC2C3DTest(48, 40, 24, 1, asdFftDirection::ASCEND_FFT_FORWARD, "_r");
}
