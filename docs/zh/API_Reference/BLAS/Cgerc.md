# Cgerc

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
asdBlasMakeCgercPlan：初始化该句柄对应的Cgerc算子配置。\
asdBlasCgerc：复数向量乘以另一个复数向量的共轭转置后，加在一个矩阵上。
- 计算公式：
  $$
  A= alpha * x *y^H + A\\ 
  $$
  示例：\
输入“x”为：\
[1.0 + 1.0j, 2.0 + 2.0j]\
输入“alpha”为：\
1.0 + 1.0j\
输入“y”为：\
[1.0 + 1.0j, 1.0 + 2.0j]\
输入“A”为：\
[ [2.0 + 3.0j, 3.0 + 4.0j], \
  [3.0 + 3.0j, 4.0 + 4.0j] ]\
调用“asdBlasCgerc”算子后，输出“A”为：\
[ [4.0 + 5.0j, 7.0 + 6.0j], \
  [7.0 + 7.0j, 12.0 + 8.0j] ]
 
## 函数原型

```Cpp
AspbStatus asdBlasMakeCgercPlan(
  asdBlasHandle handle)

```

```Cpp
AspbStatus asdBlasCgerc(
  asdBlasHandle               handle, 
  const int64_t               m, 
  const int64_t               n, 
  const std::complex<float> & alpha,
  aclTensor *                 x, 
  const int64_t               incx, 
  aclTensor *                 y, 
  const int64_t               incy, 
  aclTensor *                 A,
  const int64_t               lda)
```

## asdBlasMakeCgercPlan

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

## asdBlasCgerc

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
      <td>表示x向量中复数元素的个数，矩阵A的行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>表示y向量中复数元素的个数，矩阵A的列数。</td>
    </tr>
    <tr>
      <td>alpha（std::complex&ltfloat&gt &）</td>
      <td>输入</td>
      <td><ul><li>公式中的alpha，输入的复数标量。</li><li>数据类型支持COMPLEX64。</li></ul></td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td><ul><li>输入向量，对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m]。</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>向量x相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输入</td>
      <td><ul><li>输入向量，对应公式中的'y'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[n]。</li></ul></td>
    </tr>
    
    <tr>
      <td>incy（int64_t）</td>
      <td>输入</td>
      <td>向量y相邻元素间的内存地址偏移量（当前约束为1）。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m，n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>表示矩阵A的第一个维度大小（当前约束为m）。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 输入的元素个数m，n当前覆盖支持[1,8192]。
- 算子输入shape为[m]、[n]、[m,n]，输出shape为[m,n]。
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
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<float> alpha = std::complex<float>(1.0f, 0.0f);

    int64_t aSize = m * n;
    int64_t xSize = m;
    int64_t ySize = n;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.reserve(aSize);
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.reserve(ySize);

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData[i * n + j] = std::complex<float>(-1.0f, -1.0f);
        }
    }
    for (int64_t i = 0; i < m; i++) {
        tensorInXData[i] = std::complex<float>(1.0 + 1.0 * i, -1.0 - 1.0 * i);
    }
    for (int64_t i = 0; i < n; i++) {
        tensorInYData[i] = std::complex<float>(1.0 + 1.0 * i, -1.0 - 1.0 * i);
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), 1, m);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, n);

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> aShape = {m, n};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *inputA = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *inputADeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCgercPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgerc(handle, m, n, alpha, inputX, incx, inputY, incy, inputA, lda));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAData.data(),
        aSize * sizeof(std::complex<float>),
        inputADeviceAddr,
        aSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor A from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(inputA);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(inputADeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
