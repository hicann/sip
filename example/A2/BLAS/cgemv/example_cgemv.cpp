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

void printTensor(const std::complex<float>* tensorData, int64_t rows, int64_t cols)
{
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            std::cout << tensorData[i * cols + j] << " ";
        }
        std::cout << std::endl;
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

    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<float> alpha = std::complex<float>(1.0, 1.0);
    std::complex<float> beta = std::complex<float>(1.0, 1.0);
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    int64_t aSize = m * n;
    int64_t xSize = n;
    int64_t ySize = m;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.resize(aSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData[i * n + j] = std::complex<float>(i + 0.0, i + 0.0);
        }
    }
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < n; i++) {
        tensorInXData[i] = std::complex<float>(i + 1.0, 2 + 0.0);
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.resize(ySize);
    for (int64_t i = 0; i < m; i++) {
        tensorInYData[i] = std::complex<float>(1.0, 1.0);
    }

    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), 1, n);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, m);

    std::vector<int64_t> aShape = {m, n};
    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> yShape = {m};
    aclTensor* inputA = nullptr;
    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    void* inputADeviceAddr = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void* buffer = nullptr;
    asdBlasMakeCgemvPlan(handle, trans, m, n, inputY, incy);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgemv(handle, trans, m, n, alpha, inputA, lda, inputX, incx, beta, inputY, incy));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInYData.data(), ySize * sizeof(std::complex<float>), inputYDeviceAddr,
                      ySize * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, m);

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
