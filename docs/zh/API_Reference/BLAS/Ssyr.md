# Ssyr

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
asdBlasMakeSsyrPlan：初始化该句柄对应的Ssyr算子配置。\
asdBlasSsyr：用于计算单精度向量的外积并将结果加到一个矩阵上。
- 计算公式：

  $$
  A=alpha*x*x^T+A
  $$

    示例：
  - 输入“x”为：\
  [   [ 1 ], \
    [ 2 ] ]
  - 输入“A”为：\
  [   [ 1, 2 ], \
    [ 3, 4 ]  ]\
  - 输入“uplo”为：U\
  - 输入“n”为：2，输入“lda”为：2，输入“incx”为：1。\
  - 输入“alpha”为：2.0。\
  - 调用“asdBlasSsyr”算子后，算子更新上三角数据，输出“A”为：\
 [   [ 3, 6 ], \
    [ 3, 12 ]  ]

## 函数原型

```Cpp
AspbStatus asdBlasMakeSsyrPlan(
  asdBlasHandle handle)

```

```Cpp
AspbStatus asdBlasSsyr(
  asdBlasHandle         handle, 
  asdBlasFillMode_t     uplo, 
  const int64_t         n, 
  const float &         alpha, 
  aclTensor *           x, 
  const int64_t         incx, 
  aclTensor *           A, 
  const int64_t         lda)
```

## asdBlasMakeSsyrPlan

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

## asdBlasSsyr

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
      <td>指定参与计算的矩阵A的三角区域。<ul><li>ASDBLAS_FILL_MODE_LOWER:下三角</li><li>ASDBLAS_FILL_MODE_UPPER:上三角</li></ul></td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>向量x中的元素个数，矩阵A的行列数。</td>
    </tr>
    <tr>
      <td>alpha（float *）</td>
      <td>输入</td>
      <td>公式中的alpha，标量，向量乘积缩放因子。</td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[n]</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>x相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[n，n]</li></ul></td>
    </tr>

    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>矩阵A的每列元素的存储步长（当前约束为n）。</td>
    </tr>
    
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  - 输入的元素个数n当前支持的范围是[1,16384]。
  - 算子输入shape为[n]、[n,n]，输出shape为[n,n]。
  - 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

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

    int64_t n = 6;
    float alpha = 2.0f;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER;
    int64_t incx = 1;
    int64_t lda = n;

    const int64_t tensorXSize = 6;
    std::vector<float> tensorInXData;
    tensorInXData.reserve(tensorXSize);
    for (int i = 0; i < tensorXSize; i++) {
        tensorInXData.push_back(1.0 + i);
    }

    const int64_t tensorASize = n * n;
    std::vector<float> tensorInAData;
    tensorInAData.reserve(tensorASize);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData.push_back(2.0f + i);
        }
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "uplo = " << static_cast<int32_t>(uplo) << std::endl;
    std::cout << "------- input X -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++)
            std::cout << tensorInAData[i * n + j] << " ";
        std::cout << std::endl;
    }

    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> matAShape = {n, n};

    aclTensor *inputX = nullptr;
    aclTensor *inputA = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputADeviceAddr = nullptr;

    ret = CreateAclTensor<float>(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(tensorInAData, matAShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeSsyrPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    // A = alpha * x * x.T + A
    ASD_STATUS_CHECK(asdBlasSsyr(handle, uplo, n, alpha, inputX, incx, inputA, lda));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAData.data(),
        n * n * sizeof(float),
        inputADeviceAddr,
        n * n * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output A -------" << std::endl;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            std::cout << tensorInAData[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputA);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputADeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
