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
#include <complex>
#include <vector>
#include "interp_api.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

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
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(stream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T) * 2; // 2 : complex
    // 调用aclrtMalloc申请device侧内存
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);

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
    // 设置算子使用的device id
    int deviceId = 0;
    // （固定写法）创建执行流
    aclrtStream stream;
    Init(deviceId, &stream);

    // 创造tensor的Host侧数据
    int64_t batch = 1;
    int64_t nRs = 2;
    int64_t totalSubcarrier = 32;
    int64_t nSignal = 14;

    int64_t xSize = batch * nRs * totalSubcarrier * 2;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t coeffSize = batch * (nSignal - nRs) * nRs * 2;
    std::vector<float> coeffData;
    coeffData.resize(xSize);
    for (int64_t i = 0; i < coeffSize; i++) {
        coeffData[i] = 1;
    }

    int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier * 2;
    std::vector<float> resultData;
    resultData.resize(resultSize);
    for (int64_t i = 0; i < resultSize; i++) {
        resultData[i] = 2;
    }

    // int64_t xSize = batch * nRs * totalSubcarrier;
    // std::vector<std::complex<float>> tensorInXData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < xSize; i++) {
    //     tensorInXData[i] = std::complex<float>(i * 2, i * 2 + 1);
    // }
    // int64_t coeffSize = batch * (nSignal - nRs) * nRs;
    // std::vector<std::complex<float>> coeffData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < coeffSize; i++) {
    //     coeffData[i] = std::complex<float>(1, 1);
    // }
    // int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier;
    // std::vector<std::complex<float>> resultData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < resultSize; i++) {
    //     resultData[i] = std::complex<float>(2, 2);
    // }

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input coeff -------" << std::endl;
    for (int64_t i = 0; i < coeffSize; i++) {
        std::cout << coeffData[i] << " ";
    }
    std::cout << std::endl;

    // 创造输入/输出tensor
    aclTensor* inputX = nullptr;
    aclTensor* inputCoeff = nullptr;
    aclTensor* result = nullptr;
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    CreateAclTensor(tensorInXData, {batch, nRs, totalSubcarrier}, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64,
                    &inputX);
    CreateAclTensor(coeffData, {batch, nSignal - nRs, nRs}, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputCoeff);
    CreateAclTensor(resultData, {batch, nSignal - nRs, totalSubcarrier}, &resultDeviceAddr, aclDataType::ACL_COMPLEX64,
                    &result);

    size_t lwork = 0;
    void* buffer = nullptr;
    AsdSip::asdInterpWithCoeffGetWorkspaceSize(lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdInterpWithCoeff(inputX, inputCoeff, result, stream, buffer);
    aclrtSynchronizeStream(stream);
    // 将输出tensor的Device侧数据复制到Host侧内存上
    aclrtMemcpy(resultData.data(), resultSize * sizeof(float), resultDeviceAddr, resultSize * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < nSignal - nRs; i++) {
        for (int64_t j = 0; j < totalSubcarrier * 2; j++) {
            std::cout << resultData[i * totalSubcarrier * 2 + j] << " ";
        }
        std::cout << std::endl;
    }

    // 资源释放
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputCoeff);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // 调用算子后重置算子使用的deviceId
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
