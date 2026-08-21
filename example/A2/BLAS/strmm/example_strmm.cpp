/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
    } while (0)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char** argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    asdBlasSideMode_t side = asdBlasSideMode_t::ASDBLAS_SIDE_LEFT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;
    const int64_t m = 5;
    const int64_t n = 5;
    float alpha = 1.0;
    int64_t lda = m;
    int64_t ldb = m;
    int64_t ldc = m;

    const int64_t tensorASize = m * m;
    std::vector<float> tensorInAData(tensorASize, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < m; j++) {
            tensorInAData[m * i + j] = i;
        }
    }

    const int64_t tensorBSize = m * n;
    std::vector<float> tensorInBData(tensorBSize, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInBData[n * i + j] = i;
        }
    }

    const int64_t tensorCSize = m * n;
    std::vector<float> tensorCData(tensorCSize, 0.0);

    std::cout << "side = " << static_cast<int32_t>(side) << std::endl;
    std::cout << "uplo = " << static_cast<int32_t>(uplo) << std::endl;
    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "diag = " << static_cast<int32_t>(diag) << std::endl;

    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < m; j++)
            std::cout << tensorInAData[i * m + j] << " ";
        std::cout << std::endl;
    }

    std::cout << "------- input B -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++)
            std::cout << tensorInBData[i * n + j] << " ";
        std::cout << std::endl;
    }

    std::vector<int64_t> aShape = {tensorASize};
    std::vector<int64_t> bShape = {tensorBSize};
    std::vector<int64_t> cShape = {tensorCSize};

    aclTensor* inputA = nullptr;
    aclTensor* inputB = nullptr;
    aclTensor* outputC = nullptr;
    void* inputADeviceAddr = nullptr;
    void* inputBDeviceAddr = nullptr;
    void* outputCDeviceAddr = nullptr;

    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInBData, bShape, &inputBDeviceAddr, aclDataType::ACL_FLOAT, &inputB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorCData, cShape, &outputCDeviceAddr, aclDataType::ACL_FLOAT, &outputC);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, inputA, lda, inputB, ldb, outputC, ldc));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorCData.data(), tensorCSize * sizeof(float), outputCDeviceAddr, tensorCSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output C -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            std::cout << tensorCData[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(outputC);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputBDeviceAddr);
    aclrtFree(outputCDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
