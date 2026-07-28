# Ctrmv

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
asdBlasMakeCtrmvPlan：初始化该句柄对应的Ctrmv算子配置。\
asdBlasCtrmv：单精度复数矩阵向量乘法算子，用于计算一个复数三角矩阵与一个复数向量的乘积。
- 计算公式：
  - 矩阵向量乘积
  $$
  x = A * x
  $$
  - 转置矩阵向量乘积
  $$
  x = A^T * x
  $$
  - 共轭转置矩阵向量乘积
  $$
  x = A^H * x
  $$
  示例：\
  输入“A”为：\
  [[ 1+2i, 1+2i ],\
  [ 1+2i, 1+2i ]]\
输入“X”为：\
    [  [ 0+0i, 1+i ]  ]\
输入“uplo”为：L，输入“trans”为：N，输入“diag”为：N。\
输入“n”为：2，输入“lda”为：2，输入“incx”为：1。\
调用asdBlasCtrmv算子后，输出“x”为：\
[  [ 0+0i, -1+3i ]  ]

## 函数原型

```Cpp
AspbStatus asdBlasMakeCtrmvPlan(
  asdBlasHandle      handle,
  asdBlasFillMode_t  uplo,
  int64_t            n)
```

```Cpp
AspbStatus asdBlasCtrmv(
  asdBlasHandle      handle,
  asdBlasFillMode_t  uplo,
  asdBlasOperation_t trans,
  asdBlasDiagType_t  diag,
  const int64_t      n,
  aclTensor *        A,
  const int64_t      lda,
  aclTensor *        x,
  const int64_t      incx)
```

## asdBlasMakeCtrmvPlan

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
      <td>uplo（asdBlasFillMode_t）</td>
      <td>输入</td>
      <td>指定矩阵A的存储格式。<ul><li>ASDBLAS_FILL_MODE_LOWER:下三角</li><li>ASDBLAS_FILL_MODE_UPPER:上三角</li></ul></td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的阶数，向量x的维度。</td>
    </tr>
  </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasCtrmv

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
      <td>uplo（asdBlasFillMode_t）</td>
      <td>输入</td>
      <td>指定矩阵A的存储格式。<ul><li>ASDBLAS_FILL_MODE_LOWER:下三角</li><li>ASDBLAS_FILL_MODE_UPPER:上三角</li></ul></td>
    </tr>
    <tr>
      <td>trans（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵A是否需要转置。<ul><li>ASDBLAS_OP_N :不转置</li><li>ASDBLAS_OP_T :转置</li><li>ASDBLAS_OP_C:上共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>diag（asdBlasDiagType_t）</td>
      <td>输入</td>
      <td>指定矩阵A的对角元素处理方式。<ul><li>ASDBLAS_DIAG_NON_UNIT:一般矩阵</li><li>ASDBLAS_DIAG_UNIT:单位矩阵</li></ul></td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的阶数，向量x的维度。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>shape为[n,n]</li></ul></td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的第一个维度大小。</td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>向量x中元素的步长。</td>
    </tr>
  </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 输入的元素个数n当前覆盖支持[1,8192]。
- 算子输入shape为[n,n]、[n]，输出shape为[n]。
- 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

- **asdBlasCtrmv**

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

    int64_t n = 4;
    int64_t incx = 1;
    int64_t lda = 4;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    int64_t tensorXSize = 4;
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(tensorXSize);
    for (int64_t i = 0; i < tensorXSize; i++) {
        tensorInXData[i] = {(float)(1.0 * i), (float)(1.0 * i)};
    }

    int64_t tensorASize = n * n;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.resize(tensorASize);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData[n * i + j] = {0.0, 0.0};
        }
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i + 1; j++) {
            tensorInAData[n * i + j] = {1.0, 2.0};
        }
    }

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            std::cout << tensorInAData[n * i + j] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<int64_t> aShape = {tensorASize};
    std::vector<int64_t> xShape = {tensorXSize};
    aclTensor *inputA = nullptr;
    aclTensor *inputX = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputXDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCtrmvPlan(handle, uplo, n);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCtrmv(handle, uplo, trans, diag, n, inputA, lda, inputX, incx));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        tensorXSize * sizeof(std::complex<float>),
        inputXDeviceAddr,
        tensorXSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputX);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputXDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
