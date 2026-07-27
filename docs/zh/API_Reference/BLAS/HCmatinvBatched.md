# HCmatinvBatched

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
asdBlasMakeHCmatinvBatchedPlan：初始化该句柄对应的算子配置。\
asdBlasHCmatinvBatched：对复数矩阵进行求逆。
- 计算公式：\
计算批量复数矩阵的逆矩阵，每个复数矩阵需满足条件$A^{-1}A=I$，其中A为非奇异方阵，且A是一个n*n的输入方阵，I为单位矩阵。

  示例：\
输入“A”为：\
[2-2i ,1-i ,1-i, 1-i\
1-i, 2-2i, 1-i ,1-i\
1-i, 1-i ,2-2i ,1-i\
1-i ,1-i, 1-i, 2-2i]\

  [3-3i ,1-i, 1-i, 1-i\
1-i, 3-3i, 1-i ,1-i\
1-i, 1-i ,3-3i ,1-i\
1-i, 1-i, 1-i, 3-3i]\
输入“n”为： 4\
输入“batchSize”为：2\
调用“asdBlasHCmatinvBatched”算子后，\
输出“Ainv”为：\
[0.4+0.4i, -0.1-0.1i,-0.1-0.1i ,-0.1-0.1i\
-0.1-0.1i ,0.4+0.4i ,-0.1-0.1i ,-0.1-0.1i\
-0.1-0.1i ,-0.1-0.1i,0.4+0.4i ,-0.1-0.1i\
-0.1-0.1i ,-0.1-0.1i,-0.1-0.1i,0.4+0.4i]\

  [0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i\
-0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i\
-0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i\
-0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i]
 
## 函数原型

```Cpp
AspbStatus asdBlasMakeHCmatinvBatchedPlan(
  asdBlasHandle        handle, 
  const int64_t        n, 
  const int64_t        batchSize)
```

```Cpp
AspbStatus asdBlasHCmatinvBatched(
  asdBlasHandle                    handle,  
  const int64_t                    n, 
  aclTensor *                      A, 
  const int64_t                    lda, 
  aclTensor *                      Ainv,
  const int64_t                    lda_inv, 
  aclTensor *                      info, 
  const int64_t                    batchSize)
```

## asdBlasMakeHCmatinvBatchedPlan

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
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>复数矩阵求逆中的矩阵数量。</td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasHCmatinvBatched

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
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>单批次矩阵A的行数。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'A'。</li><li>行主序。</li><li>数据类型支持COMPLEX32。</li><li>数据格式支持ND。</li><li>shape为[batch,n,n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>A左右相邻元素间的内存地址偏移量（当前约束为n）。</td>
    </tr>
    <tr>
      <td>Ainv（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>输出的逆矩阵。</li><li>数据类型支持COMPLEX32。</li><li>数据格式支持ND。</li><li>shape为[batch, n, n]。</li></ul></td>
    </tr>
    <tr>
      <td>lda_inv（int64_t）</td>
      <td>输入</td>
      <td>输出的逆矩阵的左右相邻元素间的内存地址偏移量（当前约束为n）。</td>
    </tr>
    <tr>
      <td>info（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>每个batch矩阵的求逆结果信息。</li><li>数据类型支持int32_t。</li><li>数据格式支持ND。</li><li>shape为[batch, 1]。</li></ul></td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>复数矩阵求逆中的矩阵数量。</td>
    </tr>
    </tbody>
    </table>
    
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- lda、lda_inv、info参数在当前版本实际未启用。
- 输入参数n小于等于256。
- 输入参数batchSize小于等于3000。

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

void printTensor(const std::complex<op::fp16_t> *tensorData, int64_t batch, int64_t rows, int64_t cols)
{
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                auto data = tensorData[b * rows * cols + i * cols + j];
                std::cout << "(" << (float)data.real() << "," << (float)data.imag() << ")" << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

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

    int64_t batchSize = 3;
    int64_t n = 4;

    int64_t tensorASize = batchSize * n * n;
    std::vector<std::complex<op::fp16_t>> tensorInAData;
    std::vector<std::complex<op::fp16_t>> tensorInAinvData;
    std::vector<int32_t> tensorInInfoData;
    tensorInAData.reserve(tensorASize);
    tensorInAinvData.reserve(tensorASize);
    tensorInInfoData.reserve(batchSize);

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                if (i == j) {
                    tensorInAData[n * n * batchIdx + n * i + j] = std::complex<op::fp16_t>(2.0f + batchIdx, -2.0f - batchIdx);
                } else {
                    tensorInAData[n * n * batchIdx + n * i + j] = std::complex<op::fp16_t>(1.0f, -1.0f);
                }
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                tensorInAinvData[n * n * batchIdx + n * i + j] = std::complex<op::fp16_t>(-1.0f, -1.0f);
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        tensorInInfoData[batchIdx] = 0;
    }

    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), batchSize, n, n);
    std::cout << "------- input TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    std::vector<int64_t> aShape = {batchSize, n, n};
    std::vector<int64_t> ainvShape = {batchSize, n, n};
    std::vector<int64_t> infoShape = {batchSize};
    aclTensor *inputA = nullptr;
    aclTensor *inputAinv = nullptr;
    aclTensor *inputInfo = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputAinvDeviceAddr = nullptr;
    void *inputInfoDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX32, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInAinvData, ainvShape, &inputAinvDeviceAddr, aclDataType::ACL_COMPLEX32, &inputAinv);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInInfoData, infoShape, &inputInfoDeviceAddr, aclDataType::ACL_INT32, &inputInfo);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeHCmatinvBatchedPlan(handle, n, batchSize);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);
    asdBlasSynchronize(handle);

    ASD_STATUS_CHECK(asdBlasHCmatinvBatched(handle, n, inputA, n, inputAinv, n, inputInfo, batchSize));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAinvData.data(),
        tensorASize * sizeof(std::complex<op::fp16_t>),
        inputAinvDeviceAddr,
        tensorASize * sizeof(std::complex<op::fp16_t>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy Ainv from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInAinv -------" << std::endl;
    printTensor(tensorInAinvData.data(), batchSize, n, n);

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputAinv);
    aclDestroyTensor(inputInfo);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputAinvDeviceAddr);
    aclrtFree(inputInfoDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
