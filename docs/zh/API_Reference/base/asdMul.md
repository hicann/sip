# asdMul

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |

## 功能说明

- 接口功能：支持向量逐元素乘积（Hadamard）能力，返回一个和输入同样形状大小的复数矩阵。
- 计算公式：

  $$
  result=A \odot\ B =(A)_{ij}(B)_{ij}
  $$

    示例：

  输入“A”为：\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
输入“B”为：\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
调用asdMul算子后，输出“result”为：\
[ [ 0+2i, 0+2i ],\
  [ 0+8i, 0+8i ] ]

## 函数原型

```Cpp
AspbStatus asdMul(
  int                 n,
  const aclTensor *   x,
  const aclTensor *   y,
  aclTensor *         z,
  void *              stream,
  void *              workspace = nullptr)
```

## asdMul

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>n（int）</td>
      <td>输入</td>
      <td>表示输入的元素个数。</td>
    </tr>
    <tr>
      <td>x（const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示输入的矩阵，对应公式中的'A'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>y（const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示输入的矩阵，对应公式中的'B'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>z（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>表示输出的矩阵，对应公式中的'result'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>NPU执行流。</td>
    </tr>
    <tr>
      <td>workspace（void *）</td>
      <td>输入</td>
      <td>asdMul算子所需要的workspace。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  - 输入的元素个数n理论支持[1,9.22e+18]。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

- **mul_complex32**

```Cpp
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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

void printTensor(const std::complex<op::fp16_t> *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << "(" << (float)tensorData[i].real() << "," << (float)tensorData[i].imag() << ")" << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;

    int64_t vecSize = n;
    std::vector<std::complex<op::fp16_t>> tensorInXData;
    std::vector<std::complex<op::fp16_t>> tensorInYData;
    tensorInXData.reserve(vecSize);
    tensorInYData.reserve(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInXData.push_back({(op::fp16_t)(9.0f + i), (op::fp16_t)(100.0f + i)});
    }
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInYData.push_back({(op::fp16_t)(22.0f + i), (op::fp16_t)(33.0f * (i + 1))});
    }
    std::vector<std::complex<op::fp16_t>> tensorOutZData(
        vecSize, {(op::fp16_t)0.0f, (op::fp16_t)0.0f});

    std::cout << "------- input X -------" << std::endl;
    printTensor(tensorInXData.data(), vecSize);
    std::cout << "------- input Y -------" << std::endl;
    printTensor(tensorInYData.data(), vecSize);

    std::vector<int64_t> xShape = {vecSize};
    std::vector<int64_t> yShape = {vecSize};
    std::vector<int64_t> zShape = {vecSize};

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX32, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX32, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutZData, zShape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX32, &outputZ);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ASD_STATUS_CHECK(asdMul(n, inputX, inputY, outputZ, stream));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutZData.data(),
        vecSize * sizeof(std::complex<op::fp16_t>),
        outputZDeviceAddr,
        vecSize * sizeof(std::complex<op::fp16_t>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy z from device to host failed. ERROR: %d\n", ret); return ret);
    std::cout << "------- output Z -------" << std::endl;

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
```

- **mul_complex64**

```Cpp
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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

void printTensor(const std::complex<float> *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
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

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutZData, zShape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX64, &outputZ);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ASD_STATUS_CHECK(asdMul(n, inputX, inputY, outputZ, stream));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutZData.data(),
        vecSize * sizeof(std::complex<float>),
        outputZDeviceAddr,
        vecSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
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
```
