# Cal

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
asdBlasMakeCalPlan：初始化该句柄对应的Cal算子配置。\
asdBlasSscal：对一个实数向量进行缩放，即将向量中的每个元素乘以一个实数alpha。\
asdBlasCsscal：对一个复数向量进行缩放，即将向量中的每个元素乘以一个实数alpha。\
asdBlasCscal：对一个复数向量进行缩放，即将向量中的每个元素乘以一个复数alpha。
- 计算公式：
  - asdBlasSscal的公式

  $$
  x= alpha * x
  $$

        示例：
        输入“x”为：
        [3.0, 4.0]
        输入“alpha”为：
        2.0
        调用asdBlasSscal算子后，输出“x”为：
        [6.0, 8.0]
  - asdBlasCsscal的公式

  $$
  x= alpha * x
  $$

        示例：
        输入“x”为：
        [3.0 + 4.0j, 4.0 + 4.0j]
        输入“alpha”为：
        2.0
        调用asdBlasCsscal算子后，输出“x”为：
        [6.0 + 8.0j, 8.0 + 8.0j]
  - asdBlasCscal的公式
  $$
  x= alpha * x
  $$
        示例：
        输入“x”为：
        [3.0 + 4.0j, 4.0 + 4.0j]
        输入“alpha”为：
        3.0 + 4.0j
        调用asdBlasCscal算子后，输出“x”为：
        [-7.0 + 24.0j, -4.0 + 28.0j]

## 函数原型

```Cpp
AspbStatus asdBlasMakeCalPlan(asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasSscal(
  asdBlasHandle   handle, 
  const int64_t   n, 
  const float &   alpha, 
  aclTensor *     x, 
  const int64_t   incx)
```

```Cpp
AspbStatus asdBlasCsscal(
  asdBlasHandle   handle, 
  const int64_t   n, 
  const float &   alpha, 
  aclTensor *     x, 
  const int64_t   incx)
```

```Cpp
AspbStatus asdBlasCscal(
  asdBlasHandle               handle, 
  const int64_t               n, 
  const std::complex<float> & alpha, 
  aclTensor *                 x,
  const int64_t               incx)
```

## asdBlasMakeCalPlan

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
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasSscal & asdBlasCsscal

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
      <td>表示总的元素个数。</td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>表示输入/输出的向量，对应公式中的'x'。</li><li>数据类型支持FLOAT32（asdBlasSscal）/数据类型支持COMPLEX64（asdBlasCsscal）</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>alpha（float &）</td>
      <td>输入</td>
      <td>向量的缩放因子。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasCscal

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 930px"><colgroup>
    <col style="width: 300px">
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
      <td>表示总的元素个数。</td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>表示输入/输出的向量，对应公式中的'x'。</li><li>数据类型支持COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>alpha（const std::complex&ltfloat&gt &）</td>
      <td>输入</td>
      <td>向量的缩放因子。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 输入的元素个数当前支持[1,6.71e+06]。
- 算子输入shape为[n]，输出shape为[n]。
- 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

- **asdBlasSscal**

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

    int64_t n = 7;
    int64_t incx = 1;
    float alpha = 10.0;

    int64_t xSize = 7;
    std::vector<float> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input X -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    aclTensor *inputX = nullptr;
    void *inputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCalPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasSscal(handle, n, alpha, inputX, incx));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        xSize * sizeof(float),
        inputXDeviceAddr,
        xSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output X -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclrtFree(inputXDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **asdBlasCscal**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include <complex>
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        } else {                                                             \
            std::cout << "Execute successfully." << std::endl;               \
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

void printTensor(const std::complex<float> *tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
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

    int64_t n = 5;
    int64_t incx = 1;
    std::complex<float> alpha = (std::complex<float>){2, 3};

    int64_t xSize = 5;
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {(float)(1.0 + i), (float)(2.0 + i)};
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);

    std::vector<int64_t> xShape = {xSize};
    aclTensor *inputX = nullptr;
    void *inputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCalPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCscal(handle, n, alpha, inputX, incx));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        xSize * sizeof(std::complex<float>),
        inputXDeviceAddr,
        xSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);

    aclDestroyTensor(inputX);
    aclrtFree(inputXDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **asdBlasCsscal**

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "asdsip.h"
#include <complex>
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        } else {                                                             \
            std::cout << "Execute successfully." << std::endl;               \
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
    int incx = 1;
    float alpha = 2.0;

    int64_t xSize = 5;
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {(float)(1.0 + i), (float)(2.0 + i)};
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input TensorInX -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> xShape = {xSize};
    aclTensor *inputX = nullptr;
    void *inputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCalPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCsscal(handle, n, alpha, inputX, incx));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        xSize * sizeof(std::complex<float>),
        inputXDeviceAddr,
        xSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInX -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    aclDestroyTensor(inputX);
    aclrtFree(inputXDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
