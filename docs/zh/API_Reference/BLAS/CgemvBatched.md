# CgemvBatched

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
asdBlasMakeCgemvBatchedPlan：初始化该句柄对应的算子配置。\
asdBlasCgemvBatched：用于计算批量复数矩阵与向量的乘积。
- 计算公式：

  $$
  y= alpha * op(A)*x + beta * y\\

  $$

  其中，op表示矩阵A做“共轭转置”或者“非转置”的操作。\
  示例：\
输入“inTensorA[i]”为：\
[   [ 1+i, 1+2i ],\
    [ 1+3i, 1+4i ]  ]\
输入“x[i]”为：\
[ 1+i, 1+i ]\
输入“trans”为： N，表示矩阵A非转置。\
输入“m”为：2，输入“n”为： 2，输入“alpha”为：1+0i，“beta”为：0+0i。\
输入“lda”为： 2。\
输入“batchCount”为：2。\
调用“asdBlasCgemvBatched”算子后，输出“y[i]”为：\
[-1+5i,-5+9i]

## 函数原型

```Cpp
AspbStatus asdBlasMakeCgemvBatchedPlan(
  asdBlasHandle        handle,
  asdBlasOperation_t   trans,
  const int64_t        m)
```

```Cpp
AspbStatus asdBlasCgemvBatched(
  asdBlasHandle                    handle,
  asdBlasOperation_t               trans,
  const int64_t                    m,
  const int64_t                    n,
  const std::complex<float> &      alpha,
  aclTensor *                      A,
  const int64_t                    lda,
  aclTensor *                      x,
  const int64_t                    incx,
  const std::complex<float> &      beta,
  aclTensor *                      y,
  const int64_t                    incy,
  const int64_t                    batchCount)
```

## asdBlasMakeCgemvBatchedPlan

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
      <td>trans（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵A是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasCgemvBatched

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
      <td>trans（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵A是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的列数。</td>
    </tr>
    <tr>
      <td>alpha（std::complex&lt;float&gt; &）</td>
      <td>输入</td>
      <td>对应公式中的alpha，复数标量，用于乘以矩阵与向量乘法的乘积，当前版本alpha的取值只能为1+0i。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'A'。</li><li>行主序。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[batchCount ,m, n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda（ int64_t）</td>
      <td>输入</td>
      <td>A左右相邻元素间的内存地址偏移量（当前约束为m）。</td>
    </tr>
    <tr>
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>当“trans”取值为非转置：shape为[batchCount, n]。</li><li>当“trans”取值为共轭转置：shape为[batchCount, m]。</li></ul></td>
    </tr>
    <tr>
      <td>incx（int64_t）</td>
      <td>输入</td>
      <td>向量x的步长（当前约束为1）。</td>
    </tr>
    <tr>
      <td>beta（std::complex&lt;float&gt; &）</td>
      <td>输入</td>
      <td>对应公式中的beta，复数标量，用于乘以向量y，当前版本beta的取值只能为0+0i。</td>
    </tr>
    <tr>
      <td>y（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>对应公式中的'y'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>当“trans”取值为非转置：shape为[batchCount, m]。</li><li>当“trans”取值为共轭转置：shape为[batchCount, n]。</li></ul></td>
    </tr>
    <tr>
      <td>incy（int64_t）</td>
      <td>输入</td>
      <td>向量y的步长（当前约束为1）。</td>
    </tr>
    <tr>
      <td>batchCount（int64_t）</td>
      <td>输入</td>
      <td>批次数量。取值范围为[2 - 314496]。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                  \
    do {                                                       \
        AsdSip::AspbStatus err_ = (err);                       \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {          \
            std::cout << "Execute failed." << std::endl;       \
            exit(-1);                                          \
        } else {                                               \
            std::cout << "Execute successfully." << std::endl; \
        }                                                      \
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

void printTensor(const std::complex<float> *tensorData, int64_t batch, int64_t rows, int64_t cols)
{
    for(int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                std::cout << tensorData[b * rows * cols + i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
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

    int64_t batch = 3;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<float> alpha = std::complex<float>(1.0, 0.0);
    std::complex<float> beta = std::complex<float>(0.0, 0.0);
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    int64_t aSize = batch * m * n;
    int64_t xSize = batch * n;
    int64_t ySize = batch * m;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.resize(aSize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                tensorInAData[b * m * n + i * n + j] = std::complex<float>(i + 0.0f, i + 0.0f);
            }
        }
    }
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            tensorInXData[b * n + i] = std::complex<float>(i + 1.0f, 2.0f);
        }
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.resize(ySize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            tensorInYData[b * m + i] = std::complex<float>(1.0f, 1.0f);
        }
    }

    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), batch, m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), batch, 1, n);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), batch, 1, m);

    std::vector<int64_t> aShape = {batch, m, n};
    std::vector<int64_t> xShape = {batch, n};
    std::vector<int64_t> yShape = {batch, m};
    aclTensor *inputA = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgemvBatched(handle, trans, m, n, alpha, inputA, lda, inputX, incx, beta, inputY, incy, batch));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInYData.data(),
        ySize * sizeof(std::complex<float>),
        inputYDeviceAddr,
        ySize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), batch, 1, m);

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
