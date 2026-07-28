# Colwise_mul

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
asdBlasMakeColwiseMulPlan：初始化该句柄对应的ColwiseMul算子配置。\
asdBlasColwiseMul：复数矩阵与复数向量按列逐点乘，返回一个和输入矩阵同样形状大小的复数矩阵。
- 计算公式：\
 ![公式](../figures/colwise.png)
  示例：\
输入“A”为：\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
输入“X”为：\
[ 1+1i, 2+2i ]\
调用“asdBlasColwiseMul”算子后，输出“result”为：\
[ [ 0+2i, 0+2i ],\
  [ 0+8i, 0+8i ] ]

## 函数原型

```Cpp
AspbStatus asdBlasMakeColwiseMulPlan(
  asdBlasHandle       handle)

```

```Cpp
AspbStatus asdBlasColwiseMul(
  asdBlasHandle       handle,
  const int64_t       m,
  const int64_t       n,
  aclTensor *         mat,
  aclTensor *         vec,
  aclTensor *         result)
```

## asdBlasMakeColwiseMulPlan

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

## asdBlasColwiseMul

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
      <td>mat（aclTensor*）</td>
      <td>输入</td>
      <td><ul><li>输入向量，对应公式中的'A'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m,n]。</li></ul></td>
    </tr>
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>矩阵mat的行数，向量vec的元素个数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵mat的列数。</td>
    </tr>
    <tr>
      <td>vec（aclTensor*）</td>
      <td>输入</td>
      <td><ul><li>输入向量，对应公式中的'X'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m]。</li></ul></td>
    </tr>
    <tr>
      <td>result（aclTensor*）</td>
      <td>输出</td>
      <td><ul><li>输出向量，对应公式中的'result'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m, n]。</li></ul></td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

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

    std::vector<std::complex<float>> tensorInMatData;
    tensorInMatData.resize(matSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInMatData[n * i + j] = (std::complex<float>){2.0, -2.0};
        }
    }

    int64_t vecSize = m;
    std::vector<std::complex<float>> tensorInVecData;
    tensorInVecData.resize(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInVecData[i] = (std::complex<float>){3.0, -4.0};
    }

    int64_t resultSize = m * n;
    std::vector<std::complex<float>> resultData;
    resultData.resize(resultSize);

    std::cout << "------- input mat -------" << std::endl;
    printTensor(tensorInMatData.data(), m, n);

    std::cout << "------- input vec -------" << std::endl;
    printTensor(tensorInVecData.data(), m, 1);

    std::vector<int64_t> matShape = {m, n};
    std::vector<int64_t> vecShape = {m};
    std::vector<int64_t> resultShape = {m, n};

    aclTensor *inputMat = nullptr;
    aclTensor *inputVec = nullptr;
    aclTensor *outputResult = nullptr;
    void *inputMatDeviceAddr = nullptr;
    void *inputVecDeviceAddr = nullptr;
    void *outputResultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInMatData, matShape, &inputMatDeviceAddr, aclDataType::ACL_COMPLEX64, &inputMat);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInVecData, vecShape, &inputVecDeviceAddr, aclDataType::ACL_COMPLEX64, &inputVec);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &outputResultDeviceAddr, aclDataType::ACL_COMPLEX64, &outputResult);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeColwiseMulPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasColwiseMul(handle, m, n, inputMat, inputVec, outputResult));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);
    buffer = nullptr;

    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(std::complex<float>),
        outputResultDeviceAddr,
        resultSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    printTensor(resultData.data(), m, n);

    aclDestroyTensor(inputMat);
    aclDestroyTensor(inputVec);
    aclDestroyTensor(outputResult);
    aclrtFree(inputMatDeviceAddr);
    aclrtFree(inputVecDeviceAddr);
    aclrtFree(outputResultDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
