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

#define ASD_STATUS_CHECK(err)                                  \
    do {                                                       \
        AsdSip::AspbStatus err_ = (err);                       \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {          \
            std::cout << "Execute failed." << std::endl;       \
            return -1;                                         \
        } else {                                               \
            std::cout << "Execute successfully." << std::endl; \
        }                                                      \
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

void printTensor(std::vector<std::complex<op::fp16_t>> tensorData, int64_t batch, int64_t rows, int64_t cols)
{
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                auto data = tensorData[b * rows * cols + i * cols + j];
                std::cout << "(" << (float)data.real() << "," << (float)data.imag() << ")" << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int batch = 2;
    int m = 3;
    int k = 3;
    int n = 3;
    asdBlasOperation_t transA = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasOperation_t transB = asdBlasOperation_t::ASDBLAS_OP_N;
    std::complex<op::fp16_t> alpha = std::complex<op::fp16_t>(1.0f, 0.0f);
    std::complex<op::fp16_t> beta = std::complex<op::fp16_t>(0.0f, 0.0f);

    int64_t lda = k;
    int64_t ldb = n;
    int64_t ldc = n;

    const int64_t tensorASize = batch * m * k;
    const int64_t tensorBSize = batch * k * n;
    const int64_t tensorCSize = batch * m * n;

    std::vector<std::complex<op::fp16_t>> tensorInAData;
    tensorInAData.reserve(tensorASize);
    for (int i = 0; i < tensorASize; i++) {
        tensorInAData.push_back(std::complex<op::fp16_t>(1.0f, i + 0.0f));
    }

    std::vector<std::complex<op::fp16_t>> tensorInBData;
    tensorInBData.reserve(tensorBSize);
    for (int i = 0; i < tensorBSize; i++) {
        tensorInBData.push_back(std::complex<op::fp16_t>(1.0f, i + 0.0f));
    }

    std::vector<std::complex<op::fp16_t>> tensorInCData;
    tensorInCData.reserve(tensorCSize);
    for (int i = 0; i < tensorCSize; i++) {
        tensorInCData.push_back(std::complex<op::fp16_t>(1.0f, i + 0.0f));
    }

    std::vector<int64_t> matAShape = {batch, m, k};
    std::vector<int64_t> matBShape = {batch, k, n};
    std::vector<int64_t> matCShape = {batch, m, n};

    aclTensor* matA = nullptr;
    aclTensor* matB = nullptr;
    aclTensor* matC = nullptr;
    void* matADeviceAddr = nullptr;
    void* matBDeviceAddr = nullptr;
    void* matCDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<op::fp16_t>>(tensorInAData, matAShape, &matADeviceAddr,
                                                    aclDataType::ACL_COMPLEX32, &matA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<op::fp16_t>>(tensorInBData, matBShape, &matBDeviceAddr,
                                                    aclDataType::ACL_COMPLEX32, &matB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<op::fp16_t>>(tensorInCData, matCShape, &matCDeviceAddr,
                                                    aclDataType::ACL_COMPLEX32, &matC);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    std::cout << "alpha = " << "(" << (float)alpha.real() << "," << (float)alpha.imag() << ")" << std::endl;
    std::cout << "beta = " << "(" << (float)beta.real() << "," << (float)beta.imag() << ")" << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData, batch, m, k);
    std::cout << "------- input TensorInB -------" << std::endl;
    printTensor(tensorInBData, batch, k, n);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeHCgemmBatchedPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(
        asdBlasHCgemmBatched(handle, transA, transB, m, n, k, alpha, matA, lda, matB, ldb, beta, matC, ldc, batch));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInCData.data(), tensorCSize * sizeof(std::complex<op::fp16_t>), matCDeviceAddr,
                      tensorCSize * sizeof(std::complex<op::fp16_t>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInC -------" << std::endl;
    printTensor(tensorInCData, batch, m, n);

    aclDestroyTensor(matA);
    aclDestroyTensor(matB);
    aclDestroyTensor(matC);
    aclrtFree(matADeviceAddr);
    aclrtFree(matBDeviceAddr);
    aclrtFree(matCDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
