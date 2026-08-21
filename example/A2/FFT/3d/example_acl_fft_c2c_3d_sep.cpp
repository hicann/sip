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
#include <fstream>
#include <random>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

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

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            return -1;                                   \
        }                                                \
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
    // 固定写法，AscendCL初始化
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 创造tensor的Host侧数据
    // int batch = 2, Nfft1 = 256, Nfft2 = 256, Nfft3 = 256; // core dd
    int batch = 2, Nfft1 = 4, Nfft2 = 4, Nfft3 = 4; // core dd
    // int batch = 32, Nfft = 256;  // c2c dft
    // int batch = 32, Nfft = 8192; // c2c fftb
    // int batch = 32, Nfft = 15000; // c2c mixed
    // int batch = 32, Nfft = 32768; // c2c fftn
    // int batch = 32, Nfft = 199 * 199;  // core any
    const int64_t tensorInSize = batch * Nfft1 * Nfft2 * Nfft3;
    std::vector<int64_t> selfShape = {batch, Nfft1, Nfft2, Nfft3};
    std::vector<int64_t> outShape = {batch, Nfft1, Nfft2, Nfft3};

    std::vector<float> inputRealHostData(tensorInSize, 0);
    std::vector<float> inputImagHostData(tensorInSize, 0);
    std::vector<float> outputRealHostData(tensorInSize, 0);
    std::vector<float> outputImagHostData(tensorInSize, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < tensorInSize; i++) {
        inputRealHostData[i] = dis(gen);
        inputImagHostData[i] = dis(gen);
    }

    void* inputRealDeviceAddr = nullptr;
    void* inputImagDeviceAddr = nullptr;
    void* outputRealDeviceAddr = nullptr;
    void* outputImagDeviceAddr = nullptr;

    aclTensor* inputReal = nullptr;
    aclTensor* inputImag = nullptr;
    aclTensor* outputReal = nullptr;
    aclTensor* outputImag = nullptr;

    ret = CreateAclTensor(inputRealHostData, selfShape, &inputRealDeviceAddr, aclDataType::ACL_FLOAT, &inputReal);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputImagHostData, selfShape, &inputImagDeviceAddr, aclDataType::ACL_FLOAT, &inputImag);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputRealHostData, outShape, &outputRealDeviceAddr, aclDataType::ACL_FLOAT, &outputReal);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputImagHostData, outShape, &outputImagDeviceAddr, aclDataType::ACL_FLOAT, &outputImag);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdFftHandle handle;
    asdFftCreate(handle);

    asdFftMakePlan3D(handle, Nfft1, Nfft2, Nfft3, asdFftType::ASCEND_FFT_C2C_SEP, asdFftDirection::ASCEND_FFT_FORWARD,
                     batch);

    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void* workspaceAddr = nullptr;
    if (work_size > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdFftSetWorkspace(handle, (uint8_t*)workspaceAddr);

    asdFftSetStream(handle, stream);
    ASD_STATUS_CHECK(asdFftExecC2CSeparated(handle, inputReal, inputImag, outputReal, outputImag));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    asdFftDestroy(handle);

    auto size = GetShapeSize(outShape);
    std::vector<float> outRealData(size, 0);
    std::vector<float> outImagData(size, 0);
    std::vector<float> workspaceData(size * 2, -1);

    ret = aclrtMemcpy(outRealData.data(), outRealData.size() * sizeof(outRealData[0]), outputRealDeviceAddr,
                      size * sizeof(outRealData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

    ret = aclrtMemcpy(outImagData.data(), outImagData.size() * sizeof(outImagData[0]), outputImagDeviceAddr,
                      size * sizeof(outImagData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

    ret = aclrtMemcpy(workspaceData.data(), workspaceData.size() * sizeof(workspaceData[0]), workspaceAddr,
                      workspaceData.size() * sizeof(workspaceData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

    // 打印输出tensor值中前16个
    std::cout << "real part:" << std::endl;
    for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<float>(outRealData[i]) << "\t";
    }

    std::cout << "\nimag part:" << std::endl;
    for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<float>(outImagData[i]) << "\t";
    }

    std::cout << "\nworkspace real part:" << std::endl;
    for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<float>(workspaceData[i]) << "\t";
    }

    std::cout << "\nworkspace imag part:" << std::endl;
    for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<float>(workspaceData[i + size]) << "\t";
    }

    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputReal);
    aclDestroyTensor(inputImag);
    aclDestroyTensor(outputReal);
    aclDestroyTensor(outputImag);
    aclrtFree(inputRealDeviceAddr);
    aclrtFree(inputImagDeviceAddr);
    aclrtFree(outputRealDeviceAddr);
    aclrtFree(outputImagDeviceAddr);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
