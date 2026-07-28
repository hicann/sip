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
#include <securec.h>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using std::complex;
using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
        }                                                \
    } while (0)

#define DINTER_CORE_SIZE 528

static void genTab(float* tab, int tabSize)
{
    static float DINTER_CORE_33x16[DINTER_CORE_SIZE];
    for (int i = 0; i < DINTER_CORE_SIZE; ++i) {
        DINTER_CORE_33x16[i] = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
    }

    for (int i = 0; i < 4; i++) {
        int zeroOffset = i * 2;
        int blockOffset = i * (33 * 2) * (16 * 2 + 8);
        for (int j = 0; j < 33; j++) {
            int rowOffset_real = blockOffset + j * (16 * 2 + 8) * 2;
            int rowOffset_imag = rowOffset_real + (16 * 2 + 8);
            for (int k = 0; k < 16; k++) {
                tab[rowOffset_real + zeroOffset + k * 2] = DINTER_CORE_33x16[j * 16 + k];
                tab[rowOffset_imag + zeroOffset + k * 2 + 1] = DINTER_CORE_33x16[j * 16 + k];
            }
        }
    }
}

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

    int batch = 1;
    int signalLength = 64;
    int interpLength = signalLength;
    const int64_t tabSize = (33 * 2) * (16 * 2 + 8) * 4; // 2 ：虚实系数，8：补零，4：不补0，补2,4,6个零
    const unsigned long inSize = batch * signalLength;
    const unsigned long posSize = batch * interpLength;
    const unsigned long tabIndexSize = batch * interpLength;
    const unsigned long outSize = batch * interpLength;

    float* tabDate = new float[tabSize]();
    genTab(tabDate, tabSize);
    std::vector<float> tab(tabDate, tabDate + tabSize);
    std::vector<complex<float>> inSignal;
    inSignal.resize(inSize);
    for (long long ii = 0; ii < signalLength; ++ii) {
        inSignal[ii] = complex<float>(ii, ii);
    }
    std::vector<int32_t> intpPos;
    intpPos.resize(posSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        intpPos[ii] = ii;
    }
    std::vector<int16_t> tabIndex;
    tabIndex.resize(tabIndexSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        tabIndex[ii] = ii % 33;
    }
    std::vector<complex<float>> outSignal;
    outSignal.resize(outSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        outSignal[ii] = complex<float>(0, 0);
    }

    aclTensor* tensorIn = nullptr;
    aclTensor* tensorTab = nullptr;
    aclTensor* tensorPos = nullptr;
    aclTensor* tensorTabIndex = nullptr;
    aclTensor* tensorOut = nullptr;
    void* tensorInDeviceAddr = nullptr;
    void* tensorTabDeviceAddr = nullptr;
    void* tensorPosDeviceAddr = nullptr;
    void* tensorTabIndexDeviceAddr = nullptr;
    void* tensorOutDeviceAddr = nullptr;
    ret = CreateAclTensor(inSignal, {batch, signalLength}, &tensorInDeviceAddr, aclDataType::ACL_COMPLEX64, &tensorIn);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tab, {1, tabSize}, &tensorTabDeviceAddr, aclDataType::ACL_FLOAT, &tensorTab);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(intpPos, {batch, interpLength}, &tensorPosDeviceAddr, aclDataType::ACL_INT32, &tensorPos);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tabIndex, {batch, interpLength}, &tensorTabIndexDeviceAddr, aclDataType::ACL_INT16,
                          &tensorTabIndex);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outSignal, {batch, interpLength}, &tensorOutDeviceAddr, aclDataType::ACL_COMPLEX64,
                          &tensorOut);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    void* workspace = nullptr;
    size_t workspaceSize = 0;
    rsInterpolationBySincGetWorkspaceSize(workspaceSize);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, static_cast<int64_t>(workspaceSize), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(rsInterpolationBySinc(tensorIn, tensorTab, tensorPos, tensorTabIndex, tensorOut, 16, 32,
                                           interpLength, stream, workspace));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(outSignal.data(), outSize * sizeof(std::complex<float>), tensorOutDeviceAddr,
                      outSize * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (long long ii = 0; ii < interpLength; ++ii) {
        std::cout << outSignal[ii] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    delete[] tabDate;
    aclDestroyTensor(tensorIn);
    aclDestroyTensor(tensorPos);
    aclDestroyTensor(tensorTab);
    aclDestroyTensor(tensorTabIndex);
    aclDestroyTensor(tensorOut);
    aclrtFree(tensorInDeviceAddr);
    aclrtFree(tensorTabDeviceAddr);
    aclrtFree(tensorPosDeviceAddr);
    aclrtFree(tensorTabIndexDeviceAddr);
    aclrtFree(tensorOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
