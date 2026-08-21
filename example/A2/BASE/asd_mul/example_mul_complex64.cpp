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
#include <complex>
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

void printTensor(const std::complex<float>* tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;

    int64_t vecSize = n;
    std::vector<std::complex<float>> tensorInXData;
    std::vector<std::complex<float>> tensorInYData;
    tensorInXData.resize(vecSize);
    tensorInYData.resize(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInXData[i] = {(float)(1.0 + i), (float)(1.0 + i)};
    }
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInYData[i] = {(float)(2.0 + i), 3.0};
    }
    std::vector<std::complex<float>> tensorOutZData(vecSize, {0.0f, 0.0f});

    std::cout << "------- input X -------" << std::endl;
    printTensor(tensorInXData.data(), vecSize);
    std::cout << "------- input Y -------" << std::endl;
    printTensor(tensorInYData.data(), vecSize);

    std::vector<int64_t> xShape = {vecSize};
    std::vector<int64_t> yShape = {vecSize};
    std::vector<int64_t> zShape = {vecSize};

    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    aclTensor* outputZ = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* outputZDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutZData, zShape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX64, &outputZ);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ASD_STATUS_CHECK(asdMul(n, inputX, inputY, outputZ, stream));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutZData.data(), vecSize * sizeof(std::complex<float>), outputZDeviceAddr,
                      vecSize * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy z from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- Output -------" << std::endl;
    printTensor(tensorOutZData.data(), vecSize);
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(outputZ);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(outputZDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
