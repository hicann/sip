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

    int64_t n = 5;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER;
    float alpha = 2.0f;
    int64_t incx = 1;
    int64_t incy = 1;
    int64_t lda = 5;

    const int64_t tensorXSize = n;

    std::vector<float> tensorInXData;
    tensorInXData.reserve(tensorXSize);
    for (int i = 0; i < tensorXSize; i++) {
        tensorInXData.push_back(1.0 + i);
    }

    const int64_t tensorYSize = n;

    std::vector<float> tensorInYData;
    tensorInYData.reserve(tensorYSize);
    for (int i = 0; i < tensorYSize; i++) {
        tensorInYData.push_back(2.0f + i);
    }

    const int64_t tensorASize = n * n;

    std::vector<float> tensorInAData;
    tensorInAData.reserve(tensorYSize);
    for (int i = 0; i < tensorASize; i++) {
        tensorInAData.push_back(1.0f);
    }

    std::cout << "alpha = " << static_cast<int32_t>(alpha) << std::endl;
    std::cout << "uplo = " << static_cast<int32_t>(uplo) << std::endl;
    std::cout << "------- input X -------" << std::endl;
    for (int64_t i = 0; i < tensorXSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "------- input Y -------" << std::endl;
    for (int64_t i = 0; i < tensorYSize; i++) {
        std::cout << tensorInYData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++)
            std::cout << tensorInAData[i * n + j] << " ";
        std::cout << std::endl;
    }

    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> yShape = {n};
    std::vector<int64_t> matAShape = {n, n};

    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    aclTensor* inputA = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* inputADeviceAddr = nullptr;

    ret = CreateAclTensor<float>(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(tensorInAData, matAShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeSsyr2Plan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasSsyr2(handle, uplo, n, alpha, inputX, incx, inputY, incy, inputA, lda));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAData.data(), n * n * sizeof(float), inputADeviceAddr, n * n * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            std::cout << tensorInAData[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(inputA);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(inputADeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
