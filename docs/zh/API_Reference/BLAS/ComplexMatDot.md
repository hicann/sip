# ComplexMatDot

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
asdBlasMakeComplexMatDotPlan：初始化该句柄对应的ComplexMatDot算子配置。\
asdBlasComplexMatDot：实现两个复数矩阵对应位置逐点相乘，返回一个和输入矩阵同样形状大小的复数矩阵。
- 计算公式：

  $$
  result_{ij}= x_{ij} * y_{ij}
  $$

  示例：\
输入“matx”为：\
  [   [ 1 +  i, 1 + 2i, 1 + 3i ],\
      [ 1 + 4i, 1 + 5i, 1 + 6i ]   ]\
输入“maty”为：\
[   [ 2 +  i, 2 + 2i, 2 + 3i ],\
      [ 2 + 4i, 2 + 5i, 2 + 6i ]   ]\
调用“ComplexMatDot”算子后，输出“result”为：\
  [   [   1 +  3i,  -2 +  6i, -7 + 9i ],\
      [ -14 + 12i, -23 + 15i, -34 + 18i ]   ]

## 函数原型

```Cpp
AspbStatus asdBlasMakeComplexMatDotPlan(
  asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasComplexMatDot(
  asdBlasHandle      handle,
  const int64_t      m,
  const int64_t      n,
  aclTensor *        matx,
  aclTensor *        maty,
  aclTensor *        result)
```

## asdBlasMakeComplexMatDotPlan

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

## asdBlasComplexMatDot

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
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>矩阵行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵列数。</td>
    </tr>
    <tr>
      <td>matx（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m,n]。</li></ul></td>
    </tr>
    <tr>
      <td>maty（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'y'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m,n]。</li></ul></td>
    </tr>
    <tr>
      <td>result（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'result'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m,n]。</li></ul></td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 非原地更新情况下，输入和输出占用的内存大小应不超过NPU的GM大小。
- 原地更新情况下，输入占用的内存大小应不超过NPU的GM大小。
- 算子输入shape为[m,n]、[m,n]，输出shape为[m,n]。
- 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

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

void printTensor(const std::complex<float> *tensorData, int64_t rows, int64_t cols)
{
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            std::cout << tensorData[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t m = 3;
    int64_t n = 2;

    int64_t matSize = m * n;
    std::vector<std::complex<float>> tensorInMatXData;
    tensorInMatXData.resize(matSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInMatXData[n * i + j] = {(float)(1.0 + i), (float)(1.0 + i)};
        }
    }
    std::vector<std::complex<float>> tensorInMatYData;
    tensorInMatYData.resize(matSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInMatYData[n * i + j] = {(float)(2.0 + i), 3.0};
        }
    }

    std::cout << "------- input matX -------" << std::endl;
    printTensor(tensorInMatXData.data(), m, n);
    std::cout << "------- input matY -------" << std::endl;
    printTensor(tensorInMatYData.data(), m, n);

    std::vector<int64_t> xShape = {matSize};
    std::vector<int64_t> yShape = {matSize};

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInMatXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInMatYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeComplexMatDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasComplexMatDot(handle, m, n, inputX, inputY, inputX));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInMatXData.data(),
        matSize * sizeof(std::complex<float>),
        inputXDeviceAddr,
        matSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- matX -------" << std::endl;
    printTensor(tensorInMatXData.data(), m, n);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
