# Strmv

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
asdBlasMakeStrmvPlan：初始化该句柄对应的Strmv算子配置。\
asdBlasStrmv：单精度接口，用于计算一个三角矩阵与一个向量的矩阵乘法。
- 计算公式：\
 矩阵向量乘积：
  $$
  x = A * x
  $$

  转置矩阵向量乘积：
  $$
  x = A^T * x
  $$

  示例：\
  输入“A”为：\
 [   [ 1, 2 ],
    [ 3, 4 ]  ]\
  输入“x”为：\
 [1,2]\
  输入“uplo”为：U，输入“trans”为：T，输入“diag”为：N。\
输入“n”为：2，输入“lda”为：2，输入“incx”为：1。\
调用“asdBlasStrmv”算子后，输出“x”为：\
[7,10]

## 函数原型

```Cpp
AspbStatus asdBlasMakeStrmvPlan(
  asdBlasHandle           handle,
  asdBlasFillMode_t       uplo,
  asdBlasOperation_t      trans,
  int64_t                 n)

```

```Cpp
AspbStatus asdBlasStrmv(
  asdBlasHandle         handle,
  asdBlasFillMode_t     uplo,
  asdBlasOperation_t    trans,
  asdBlasDiagType_t     diag,
  const int64_t         n,
  aclTensor*            A,
  const int64_t         lda,
  aclTensor*            x,
  const int64_t         incx)
```

## asdBlasMakeStrmvPlan

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
      <td>指定是否对矩阵A进行转置。<ul><li>ASDBLAS_OP_N:不转置</li><li>ASDBLAS_OP_T:转置</li></ul></td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的行数和列数，向量x的元素个数。</td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasStrmv

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
      <td>diag（asdBlasDiagType_t）</td>
      <td>输入</td>
      <td>指定是否假定矩阵A的对角线元素为1。<ul><li>ASDBLAS_DIAG_NON_UNIT:不假定为1</li><li>ASDBLAS_DIAG_UNIT:假定为1</li></ul></td>
    </tr>
    <tr>
      <td>trans（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定是否对矩阵A进行转置。<ul><li>ASDBLAS_OP_N:不转置</li><li>ASDBLAS_OP_T:转置</li></ul></td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的行数和列数，向量x的元素个数。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[n，n]</li></ul></td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>表示张量A中元素的间隔。</td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>表示向量x中元素的间隔。</td>
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

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担责任。

- **asdBlasStrmv**

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

void printTensor(std::vector<std::complex<float>> tensorData, int64_t tensorSize)
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

    int64_t n = 4;
    int64_t incx = 1;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    const int64_t tensorASize = n * n;
    const int64_t tensorXSize = n;

    std::vector<float> tensorInAData;
    tensorInAData.resize(tensorASize);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i + 1; j++) {
            tensorInAData[n * i + j] = 1.0 + i * n + j;
        }
        for (int64_t j = i + 1; j < n; j++) {
            tensorInAData[n * i + j] = 0.0;
        }
    }

    std::vector<float> tensorInXData;
    tensorInXData.resize(tensorXSize);
    for (int64_t i = 0; i < n; i++) {
        tensorInXData[i] = 1.0;
    }

    std::cout << "uplo = " << static_cast<int32_t>(uplo) << std::endl;
    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "diag = " << static_cast<int32_t>(diag) << std::endl;
    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++)
            std::cout << tensorInAData[i * n + j] << " ";
        std::cout << std::endl;
    }
    std::cout << "------- input X -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int64_t> aShape = {tensorASize};
    std::vector<int64_t> xShape = {tensorXSize};

    aclTensor *inputA = nullptr;
    aclTensor *inputX = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputXDeviceAddr = nullptr;

    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeStrmvPlan(handle, uplo, trans, n);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasStrmv(handle, uplo, trans, diag, n, inputA, n, inputX, incx));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        tensorXSize * sizeof(float),
        inputXDeviceAddr,
        tensorXSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output X -------" << std::endl;
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
