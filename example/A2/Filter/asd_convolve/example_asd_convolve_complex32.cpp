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
#include "asdsip.h"
#include "filter_api.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl_meta.h"
#include <complex>
#include <vector>

using namespace AsdSip;

using half = op::fp16_t;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ACL_SUCCESS) {               \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
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

template <typename T>
void printTensor(std::vector<T> tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
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

    int64_t signalLen = 128; // 26208
    int64_t kernelLen = 32;
    int64_t batchCount = 2; // 768

    std::vector<std::complex<half>> tensorSignalData;
    tensorSignalData.resize(signalLen * batchCount);

    std::vector<half> tensorKernelData;
    tensorKernelData.resize(kernelLen);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorSignalData[i] = {(half)1.0, (half)1.0};
    }

    for (int64_t i = 0; i < kernelLen; i++) {
        tensorKernelData[i] = (half)(1.0 + i);
        // tensorKernelData[i] = 1.0;
    }

    std::vector<std::complex<half>> tensorOutData;
    tensorOutData.resize(signalLen * batchCount);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorOutData[i] = {(half)-1.0, (half)-1.0};
    }

    std::vector<int64_t> signalShape = {batchCount, signalLen};
    std::vector<int64_t> kernelShape = {kernelLen};
    std::vector<int64_t> resultShape = {batchCount, signalLen};

    aclTensor* signal = nullptr;
    aclTensor* kernel = nullptr;
    aclTensor* output = nullptr;
    void* signalDeviceAddr = nullptr;
    void* kernelDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<half>>(tensorSignalData, signalShape, &signalDeviceAddr,
                                              aclDataType::ACL_COMPLEX32, &signal);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<half>(tensorKernelData, kernelShape, &kernelDeviceAddr, aclDataType::ACL_FLOAT16, &kernel);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<half>>(tensorOutData, resultShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX32,
                                              &output);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t lwork = 0;
    AsdSip::asdConvolveGetWorkspaceSize(signalLen, kernelLen, lwork);
    void* buffer = nullptr;

    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(AsdSip::asdConvolve(signal, kernel, output, asdConvolveMode_t::ASD_CONVOLVE_SAME, stream, buffer));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutData.data(), signalLen * batchCount * sizeof(std::complex<half>), outputDeviceAddr,
                      signalLen * batchCount * sizeof(std::complex<half>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int batchIdx = 0; batchIdx < batchCount; batchIdx++) {
        for (int i = 0; i < signalLen; i++) {
            std::cout << "(" << (float)tensorOutData[batchIdx * signalLen + i].real() << ","
                      << (float)tensorOutData[batchIdx * signalLen + i].imag() << ")"
                      << " ";
        }
        std::cout << std::endl;
    }

    aclDestroyTensor(signal);
    aclDestroyTensor(kernel);
    aclDestroyTensor(output);
    aclrtFree(signalDeviceAddr);
    aclrtFree(kernelDeviceAddr);
    aclrtFree(outputDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
