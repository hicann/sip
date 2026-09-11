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
#include <random>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;

std::string GetSsyrOutputDirectory()
{
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}

float GetRandomAlpha()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(0.5f, 2.0f);
    return distr(gen);
}

void RunSsyrTest(asdBlasFillMode_t uplo, const std::string& uploStr)
{
    std::string originPath = GetSsyrOutputDirectory() + "/tests/ut/unittest/blas/ssyr/ssyr_data";
    std::string destPath = GetSsyrOutputDirectory() + "/build/tests/ut/unittest/blas/ssyr";
    std::string mkdir_cmd = "mkdir -p " + ShellQuote(destPath);
    std::string copy_cmd = "cp -rf " + ShellQuote(originPath) + " " + ShellQuote(destPath);
    std::string chmod_cmd = "chmod -R 755 " + ShellQuote(destPath + "/ssyr_data/");
    system(mkdir_cmd.c_str());
    system(copy_cmd.c_str());
    system(chmod_cmd.c_str());

    int deviceId = 0;
    int64_t n = 6;
    int64_t incx = 1;
    int64_t lda = n;
    float alpha = GetRandomAlpha();

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n, n}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensorX = context.inTensors[0];
    Tensor inputTensorA = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) +
                               ".bin";
        if (SaveTensorToBin(context.inTensors[i], destPath + "/ssyr_data/" + filename)) {
            std::cout << "Tensor saved successfully: " << filename << std::endl;
        } else {
            std::cerr << "Failed to save tensor: " << filename << std::endl;
        }
    }

    std::string gen_data_cmd = "cd " + ShellQuote(destPath + "/ssyr_data/") + " && python3 gen_data.py " +
                               std::to_string(alpha) + " " + uploStr;
    system(gen_data_cmd.c_str());

    aclTensor* aclInputX = nullptr;
    aclTensor* aclInputA = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputADeviceAddr = nullptr;

    std::vector<float> inputXHostData(n);
    float* inputXDataPtr = static_cast<float*>(inputTensorX.hostData);
    for (int64_t i = 0; i < n; i++) {
        inputXHostData[i] = inputXDataPtr[i];
    }

    std::vector<float> inputAHostData(n * n);
    float* inputADataPtr = static_cast<float*>(inputTensorA.hostData);
    for (int64_t i = 0; i < n * n; i++) {
        inputAHostData[i] = inputADataPtr[i];
    }

    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> aShape = {n, n};

    auto ret = CreateAclTensor(inputXHostData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &aclInputX);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(inputAHostData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &aclInputA);
    ASSERT_EQ(ret, 0);

    asdBlasHandle handle;
    AspbStatus blasStatus = asdBlasCreate(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasMakeSsyrPlan(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    blasStatus = asdBlasGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void* workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        blasStatus = asdBlasSetWorkspace(handle, workspaceAddr);
        ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    blasStatus = asdBlasSetStream(handle, aclStream);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasSsyr(handle, uplo, n, alpha, aclInputX, incx, aclInputA, lda);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    blasStatus = asdBlasSynchronize(handle);
    ASSERT_EQ(blasStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdBlasDestroy(handle);

    ret = aclrtMemcpy(inputTensorA.hostData, inputTensorA.dataSize, inputADeviceAddr, inputTensorA.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);

    std::string outFilename = GetElementDtype(inputTensorA.desc.dtype) + "_output_.bin";
    if (SaveOutTensorToBin(inputTensorA, destPath + "/ssyr_data/" + outFilename)) {
        std::cout << "Output tensor saved successfully" << std::endl;
    } else {
        std::cerr << "Failed to save output tensor" << std::endl;
    }

    aclDestroyTensor(aclInputX);
    aclDestroyTensor(aclInputA);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputADeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + ShellQuote(destPath + "/ssyr_data/") + " && python3 compare_data.py";
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}
} // namespace

TEST(TestBlasSsyr, TestSsyrLower) { RunSsyrTest(asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, "lower"); }

TEST(TestBlasSsyr, TestSsyrUpper) { RunSsyrTest(asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER, "upper"); }

// 负向校验（issue #130）：A 元素数不足 n*n 时应被拦截，而非 kernel 越界写
TEST(TestBlasSsyr, TestSsyrRejectInvalidAShape)
{
    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);

    const int64_t n = 4;
    std::vector<float> xHost(n, 1.0f);
    std::vector<float> aHost(n * n - 8, 1.0f); // 元素数 8 < n*n=16
    aclTensor* aclX = nullptr;
    aclTensor* aclA = nullptr;
    void* xAddr = nullptr;
    void* aAddr = nullptr;
    ASSERT_EQ(CreateAclTensor(xHost, std::vector<int64_t>{n}, &xAddr, aclDataType::ACL_FLOAT, &aclX), 0);
    ASSERT_EQ(CreateAclTensor(aHost, std::vector<int64_t>{n * n - 8}, &aAddr, aclDataType::ACL_FLOAT, &aclA), 0);

    asdBlasHandle handle;
    ASSERT_EQ(asdBlasCreate(handle), AsdSip::ErrorType::ACL_SUCCESS);
    ASSERT_EQ(asdBlasMakeSsyrPlan(handle), AsdSip::ErrorType::ACL_SUCCESS);

    EXPECT_EQ(asdBlasSsyr(handle, asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER, n, 1.0f, aclX, 1, aclA, n),
              AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH);

    asdBlasDestroy(handle);
    aclDestroyTensor(aclX);
    aclDestroyTensor(aclA);
    aclrtFree(xAddr);
    aclrtFree(aAddr);

    TensorContext context;
    OpTestEnd(deviceId, context, stream);
}
