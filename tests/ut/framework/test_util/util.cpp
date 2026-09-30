/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_UTIL_H
#define ASDOPS_UTIL_H

#include <array>
#include <cfloat>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

#include "utils/op_desc.h"
#include "mki/operation.h"
#include "mki/utils/any/any.h"
#include "mki/utils/SVector/SVector.h"
#include "mki/utils/fp16/fp16_t.h"
#include <fstream>
#include <iterator>
#include <random>
#include <securec.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "ops.h"
#include "mki/utils/rt/rt.h"
#include "log/log.h"
#include "utils/assert.h"
#include "utils/aspb_status.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include <unordered_map>

using namespace AsdSip;
using namespace Mki;

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

namespace AsdSip {

constexpr float FLOAT_MIN = -300;
constexpr float FLOAT_MAX = 300;

static std::runtime_error MakeRuntimeError(const std::string& message)
{
    ASDSIP_LOG(ERROR) << message;
    return std::runtime_error(message);
}

struct TensorContext {
    OpDesc opDesc;
    SVector<Tensor> inTensors;
    SVector<Tensor> outTensors;
};

struct aclTensorContext {
    OpDesc opDesc;
    SVector<Tensor> inTensors;
    SVector<Tensor> outTensors;
    SVector<aclTensor*> aclInTensors;
    SVector<aclTensor*> aclOutTensors;
};

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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

/**
 * @brief 获取测试实际使用的 deviceId: 优先读取环境变量 ASDSIP_DEVICE_ID
 *        (build.sh 按 CI 映射卡解析出的物理卡号注入, 值可能为列表如 "4,5", 取首个编号),
 *        未设置时沿用用例传入的默认值(0)。
 *        注: UT 进程启动时已剥离 ASCEND_RT_VISIBLE_DEVICES, 无可见性限制时物理卡号
 *        即进程内合法 id, 不依赖运行时对可见卡的重编号语义。
 */
static int OpTestGetDeviceId(int deviceId)
{
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId == nullptr || envDeviceId[0] == '\0') {
        return deviceId;
    }
    return std::atoi(envDeviceId);
}

/**
 * @brief 设置deviceid并创造stream.
 */
static MkiRtStream OpTestInit(int deviceId)
{
    deviceId = OpTestGetDeviceId(deviceId);
    ASDSIP_LOG(INFO) << "MkiRtDeviceSetCurrent " << deviceId;
    int ret = MkiRtDeviceSetCurrent(deviceId);
    if (ret != 0) {
        std::stringstream ss;
        ss << "MkiRtDeviceSetCurrent failed, deviceId: " << deviceId << ", ret: " << ret;
        throw MakeRuntimeError(ss.str());
    }

    MkiRtStream stream = nullptr;
    ASDSIP_LOG(INFO) << "MkiRtStreamCreate call";
    ret = MkiRtStreamCreate(&stream, 0);
    if (ret != 0 || stream == nullptr) {
        std::stringstream ss;
        ss << "MkiRtStreamCreate failed, ret: " << ret << ", stream: " << stream;
        throw MakeRuntimeError(ss.str());
    }
    return stream;
}

/**
 * @brief 设置deviceid并创造stream.
 */
static int OpTestAclInit(int deviceId, aclrtStream* stream)
{
    deviceId = OpTestGetDeviceId(deviceId);
    // auto ret = aclInit(nullptr);
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

/**
 * @brief 清空释放goldenContext_的tensor数据，并重置launchParam和runInfo。
 */
static void OpTestCleanup(TensorContext& tensorContext, MkiRtStream& stream)
{
    for (auto tensor : tensorContext.inTensors) {
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    tensorContext.inTensors.clear();

    for (auto tensor : tensorContext.outTensors) {
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    tensorContext.outTensors.clear();

    if (stream) {
        MkiRtStreamDestroy(stream);
    }
}

static void OpTestCleanup(aclTensorContext& tensorContext)
{
    for (auto tensor : tensorContext.inTensors) {
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    tensorContext.inTensors.clear();

    for (auto tensor : tensorContext.outTensors) {
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    tensorContext.outTensors.clear();
}

static void OpTestEnd(int deviceId, TensorContext& tensorContext, MkiRtStream& stream)
{
    deviceId = OpTestGetDeviceId(deviceId);
    OpTestCleanup(tensorContext, stream);
    int ret = MkiRtDeviceResetCurrent(deviceId);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "MkiRtDeviceResetCurrent fail";
}

static void OpTestEnd(int deviceId, aclTensorContext& tensorContext, MkiRtStream& stream)
{
    deviceId = OpTestGetDeviceId(deviceId);
    OpTestCleanup(tensorContext);
    int ret = aclrtDestroyStream(stream);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtDestroyStream fail";
    ret = aclrtResetDevice(deviceId);
    ASDSIP_LOG_IF(ret != 0, ERROR) << "aclrtResetDevice fail";
}

// hostdata copy to device data
static Status OpTestHostTensor2DeviceTensor(Tensor& tensor)
{
    int st = MkiRtMemMallocDevice(&tensor.data, tensor.dataSize, MKIRT_MEM_DEFAULT);
    if (st != MKIRT_SUCCESS) {
        std::stringstream ss;
        ss << "Device malloc intensor failed, ret: " << st << ", dataSize: " << tensor.dataSize
           << ", dtype: " << tensor.desc.dtype << ", dims: " << tensor.desc.dims;
        ASDSIP_LOG(ERROR) << ss.str();
        return Status::FailStatus(-1, ss.str());
    }

    st = MkiRtMemCopy(tensor.data, tensor.dataSize, tensor.hostData, tensor.dataSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        std::stringstream ss;
        ss << "Memcpy host to device failed, ret: " << st << ", dataSize: " << tensor.dataSize
           << ", dtype: " << tensor.desc.dtype << ", dims: " << tensor.desc.dims;
        ASDSIP_LOG(ERROR) << ss.str();
        return Status::FailStatus(-1, ss.str());
    }
    return Status::OkStatus();
}

/**
 * @brief 随机生成tensor的hostdata值
 * @param tensorDesc tensor描述信息
 */
static Tensor OpTestCreateHostRandTensor(const TensorDesc& tensorDesc)
{
    Tensor tensor;
    tensor.desc = tensorDesc;

    std::default_random_engine eng(42); // 使用固定种子保证测试可复现
    if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
        float* tensorData = static_cast<float*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<float>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_COMPLEX64) {
        tensor.dataSize = tensor.Numel() * sizeof(float) * 2;
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
        std::complex<float>* tensorData = static_cast<std::complex<float>*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = {static_cast<float>(distr(eng)), static_cast<float>(distr(eng))};
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(fp16_t);
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
        fp16_t* tensorData = static_cast<fp16_t*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<fp16_t>(distr(eng));
        }
        ASDSIP_LOG(INFO) << "create fp16 call!";
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(int32_t);
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
        int32_t* tensorData = static_cast<int32_t*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int32_t>(distr(eng));
        }
        ASDSIP_LOG(INFO) << "create int32 call!";
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT16) {
        tensor.dataSize = tensor.Numel() * sizeof(int16_t);
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
        int16_t* tensorData = static_cast<int16_t*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int16_t>(distr(eng));
        }
        ASDSIP_LOG(INFO) << "create int16 call!";
    } else if (tensorDesc.dtype == TENSOR_DTYPE_COMPLEX32) {
        tensor.dataSize = tensor.Numel() * sizeof(fp16_t) * 2;
        tensor.hostData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(0, 1);
        fp16_t* tensorData = static_cast<fp16_t*>(tensor.hostData);
        for (int64_t i = 0; i < tensor.Numel() * 2; i++) {
            tensorData[i] = static_cast<fp16_t>(distr(eng));
        }
    } else {
        std::stringstream ss;
        ss << "dtype not support in CreateHostRandTensor, dtype: " << tensorDesc.dtype << ", dims: " << tensorDesc.dims;
        throw MakeRuntimeError(ss.str());
    }

    if (tensor.dataSize > 0 && tensor.hostData == nullptr) {
        std::stringstream ss;
        ss << "Host malloc intensor failed, dataSize: " << tensor.dataSize << ", dtype: " << tensorDesc.dtype
           << ", dims: " << tensorDesc.dims;
        throw MakeRuntimeError(ss.str());
    }
    return tensor;
}

/**
 * @brief tesor的device侧值复制到host侧
 * @param tensor tensor
 */
static Status CopyDeviceTensorToHostTensor(Tensor& tensor)
{
    if (tensor.hostData == nullptr) {
        tensor.hostData = malloc(tensor.dataSize);
        if (tensor.hostData == nullptr) {
            return Status::FailStatus(-1, "Host malloc outtensor failed.");
        }
    }

    int st = MkiRtMemCopy(tensor.hostData, tensor.dataSize, tensor.data, tensor.dataSize, MKIRT_MEMCOPY_DEVICE_TO_HOST);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "MkiRtMemCopy";
        return Status::FailStatus(-1, "Memcpy outtensor device to host failed");
    }

    return Status::OkStatus();
}

// 辅助函数：根据dtype返回元素大小（字节）
inline size_t GetElementSize(TensorDType dtype)
{
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT:
            return 4;
        case TENSOR_DTYPE_FLOAT16:
            return 2;
        case TENSOR_DTYPE_INT16:
            return 2;
        case TENSOR_DTYPE_INT32:
            return 4;
        case TENSOR_DTYPE_INT64:
            return 8;
        case TENSOR_DTYPE_COMPLEX64:
            return 8;
        case TENSOR_DTYPE_COMPLEX32:
            return 4; // 2 * float16 = 2 * 2 bytes
        case TENSOR_DTYPE_UINT8:
            return 1;
        // 添加其他需要的类型
        default:
            return 0; // 未知类型
    }
}

// 辅助函数：根据dtype返回元素类型字符串
inline std::string GetElementDtype(TensorDType dtype)
{
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT:
            return "float";
        case TENSOR_DTYPE_FLOAT16:
            return "half";
        case TENSOR_DTYPE_INT16:
            return "int16";
        case TENSOR_DTYPE_INT32:
            return "int32";
        case TENSOR_DTYPE_INT64:
            return "int64";
        case TENSOR_DTYPE_COMPLEX64:
            return "complex64";
        case TENSOR_DTYPE_COMPLEX32:
            return "complex32";
        case TENSOR_DTYPE_UINT8:
            return "uint5";
        // 添加其他需要的类型
        default:
            return "unknown"; // 未知类型
    }
}

inline bool SaveTensorToBin(const Tensor& tensor, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        return false;

    // 1. 写入dtype (int32_t)
    int32_t dtype = static_cast<int32_t>(tensor.desc.dtype);
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

    // 2. 写入维度数量 (int32_t)
    int32_t dimCount = static_cast<int32_t>(tensor.desc.dims.size());
    file.write(reinterpret_cast<const char*>(&dimCount), sizeof(dimCount));

    // 3. 写入每个维度值 (int64_t)
    for (int64_t dim : tensor.desc.dims) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    // 4. 写入原始数据
    size_t elemSize = GetElementSize(tensor.desc.dtype);
    int64_t numElements = tensor.Numel();
    size_t dataByteSize = numElements * elemSize;

    if (tensor.hostData && dataByteSize > 0) {
        file.write(reinterpret_cast<const char*>(tensor.hostData), dataByteSize);
    }

    return !file.fail();
}

inline bool SaveOutTensorToBin(const Tensor& tensor, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        return false;

    // 4. 写入原始数据
    size_t elemSize = GetElementSize(tensor.desc.dtype);
    int64_t numElements = tensor.Numel();
    size_t dataByteSize = numElements * elemSize;

    if (tensor.hostData && dataByteSize > 0) {
        file.write(reinterpret_cast<const char*>(tensor.hostData), dataByteSize);
    }

    return !file.fail();
}

/**
 * @brief 根据tensor描述信息生成tensor
 * @param inTensorDescs tensors描述信息集
 * @param TensorContext 记录input和output的tensor的数据结构
 */
static Status OpTestMallocInTensors(const SVector<TensorDesc>& inTensorDescs, TensorContext& tensorContext)
{
    size_t tensorIdx = 0;
    for (auto tensorDesc : inTensorDescs) {
        Tensor tensor;
        tensor = OpTestCreateHostRandTensor(tensorDesc);
        Status status = OpTestHostTensor2DeviceTensor(tensor);
        if (!status.Ok()) {
            std::stringstream ss;
            ss << "OpTestMallocInTensors failed at input index " << tensorIdx << ": " << status.Message();
            if (tensor.data) {
                MkiRtMemFreeDevice(tensor.data);
            }
            if (tensor.hostData) {
                free(tensor.hostData);
            }
            for (auto createdTensor : tensorContext.inTensors) {
                if (createdTensor.data) {
                    MkiRtMemFreeDevice(createdTensor.data);
                }
                if (createdTensor.hostData) {
                    free(createdTensor.hostData);
                }
            }
            tensorContext.inTensors.clear();
            throw MakeRuntimeError(ss.str());
        }
        tensorContext.inTensors.push_back(tensor);
        ++tensorIdx;
    }
    return Status::OkStatus();
}

static Status toAclTensorForUT(const Tensor& inTensor, aclTensor** outTensor, TensorDesc desc)
{
    SVector<int64_t> shape = desc.dims;

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *outTensor = aclCreateTensor(shape.data(), shape.size(), (aclDataType)inTensor.desc.dtype, strides.data(), 0,
                                 (aclFormat)inTensor.desc.format, shape.data(), shape.size(), inTensor.data);
    return Status::OkStatus();
}

static Status OpTestMallocInAclTensors(const SVector<TensorDesc>& inTensorDescs, aclTensorContext& tensorContext)
{
    for (auto tensorDesc : inTensorDescs) {
        Tensor tensor;
        tensor = OpTestCreateHostRandTensor(tensorDesc);
        Status status = OpTestHostTensor2DeviceTensor(tensor);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << status.Message();
            return Status::FailStatus(-1, status.Message());
        }

        aclTensor* curr = nullptr;
        toAclTensorForUT(tensor, &curr, tensorDesc);

        tensorContext.inTensors.push_back(tensor);
        tensorContext.aclInTensors.push_back(curr);
    }
    return Status::OkStatus();
}

} // namespace AsdSip

#endif
