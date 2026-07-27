# Nrm2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |

## 功能说明

- 接口功能：\
asdBlasMakeNrm2Plan：初始化该句柄对应的Nrm2算子配置。\
asdBlasSnrm2：用于计算实数向量的欧氏范数。\
asdBlasScnrm2：用于计算复数向量的欧氏范数。
- 计算公式：
  - asdBlasSnrm2的公式

  $$
  result=||x||_{2}= {\sqrt{\sum _{i=1}^n|x_{i}|^2}}
  $$
    其中$x_{i}$表示向量x中的第i个元素。

        示例：
        输入“x”为：
        [1, 2, -3, 4]
        调用asdBlasSnrm2算子后，输出“result”为：
        5.47723
  - asdBlasScnrm2的公式

  $$
  result=||x||_{2}= {\sqrt{\sum _{i=1}^n|x_{i}|^2}}
  $$
  其中，$x_{i}$表示向量x中的第i个元素，$x_{i}$是复数元素。
$$
|x_{i}|^2=x_{i}\_real^2+x_{i}\_image^2
$$

         示例：
         输入“x”为：
         [1+2i, 2-2i, -3+3i, 4-3i]
         调用asdBlasScnrm2算子后，输出“result”为：
         7.48331

## 函数原型

```Cpp
AspbStatus asdBlasMakeNrm2Plan(
  asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasSnrm2(
  asdBlasHandle       handle, 
  const int64_t       n, 
  aclTensor *         x, 
  const int64_t       incx, 
  aclTensor *         result)
```

```Cpp
AspbStatus asdBlasScnrm2(
  asdBlasHandle       handle, 
  const int64_t       n, 
  aclTensor *         x, 
  const int64_t       incx, 
  aclTensor *         result)
```

## asdBlasMakeNrm2Plan

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
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>算子的句柄</td>
    </tr>
  </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasSnrm2 & asdBlasScnrm2

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
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>算子的句柄。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>总的元素个数。</td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>asdBlasSnrm2支持的数据类型为FLOAT32。</li><li>asdBlasScnrm2支持的数据类型为COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[n]。</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>result（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>表示输出的结果，对应公式中的'result'。</li><li>数据类型支持FLOAT32，只包含一个元素。</li><li>数据格式支持ND。</li><li>shape为[1]。</li></ul></td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  - 输入的元素个数n，当前覆盖支持[1,6.71e+06]。
  - 算子输入shape为[n]，输出shape为[1]。
  - 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

- **asdBlasSnrm2**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;
    int64_t incx = 1;

    int64_t xSize = 8;
    int64_t ySize = 1;
    std::vector<float> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    std::vector<float> tensorOutYData;
    tensorOutYData.reserve(ySize);

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutYData, yShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeNrm2Plan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasSnrm2(handle, n, inputX, incx, outputY));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorOutYData.data(),
        ySize * sizeof(float),
        outputYDeviceAddr,
        ySize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    std::cout << tensorOutYData[0] << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(outputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **asdBlasScnrm2**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 5;
    int64_t incx = 1;
    int64_t xSize = 5;
    int64_t ySize = 1;

    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {(float)(2.0 + i), (float)(3.0 + i)};
    }

    std::vector<float> tensorOutYData;
    tensorOutYData.reserve(ySize);

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutYData, yShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeNrm2Plan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasScnrm2(handle, n, inputX, incx, outputY));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorOutYData.data(),
        ySize * sizeof(float),
        outputYDeviceAddr,
        ySize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "=== result ===" << std::endl;
    std::cout << tensorOutYData[0] << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(outputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
