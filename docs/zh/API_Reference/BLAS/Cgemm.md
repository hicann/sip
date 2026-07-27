# Cgemm

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
asdBlasMakeCgemmPlan：初始化该句柄对应的Cgemm算子配置。\
asdBlasCgemm：矩阵乘法运算中的一种，用于计算两个复数矩阵的乘积。
- 计算公式：
  $$
  C= alpha * op(A)*op(B) + beta * C\\
  其中，op(X)= X \ \ \  \text{or} \ \ \  op(X) = X^T \ \ \  \text{or} \ \ \   op(X) = X^H 
  $$
  示例：\
输入“inTensorA”为：\
[   [ 1+i, 1+2i ],
    [ 1+3i, 1+4i ]  ]\
输入“inTensorB”为：\
[   [ 2+i, 2+2i ],
    [ 2+3i, 2+4i ]  ]\
输入“inTensorC”为：\
[   [ 3+i, 3+2i ],
    [ 3+3i, 3+4i ]  ]\
输入“transa”为：N，输入“transb”为：T。\
输入“m”为：2，输入“n”为： 2，输入“k”为：2，输入“alpha”为：1+i，“beta”为：2+2i。\
输入“lda”为： 2，输入“ldb”为：2，输入“ldc”为：2。\
调用“Cgemm”算子后，输出“outTensor”为：\
[   [ -6+16i, -18+16i ],
    [ -24+22i, -44+14i ]  ]
 
## 函数原型

```Cpp
AspbStatus asdBlasMakeCgemmPlan(
  asdBlasHandle      handle, 
  asdBlasOperation_t transa, 
  asdBlasOperation_t transb, 
  int64_t            m,
  int64_t            n, 
  int64_t            k, 
  int64_t            lda, 
  int64_t            ldb, 
  int64_t            ldc)
```

```Cpp
AspbStatus asdBlasCgemm(
  asdBlasHandle             handle, 
  asdBlasOperation_t        transa, 
  asdBlasOperation_t        transb, 
  const int64_t             m,
  const int64_t             n, 
  const int64_t             k, 
  const std::complex<float> *alpha, 
  aclTensor *               A,
  const int64_t             lda, 
  aclTensor *               B, 
  const int64_t             ldb, 
  const std::complex<float> *beta,
  aclTensor *               C, 
  const int64_t             ldc)
```

## asdBlasMakeCgemmPlan

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
      <td>transa（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵A是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>transb（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵B是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>矩阵A和C的行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵B和C的列数。</td>
    </tr>
    <tr>
      <td>k（int64_t）</td>
      <td>输入</td>
      <td>矩阵A和B的公共维度。</td>
    </tr>
    <tr>
      <td>lda（ int64_t）</td>
      <td>输入</td>
      <td>矩阵A左右相邻元素间的内存地址偏移量（当前约束为m）。</td>
    </tr>
    <tr>
      <td>ldb（ int64_t）</td>
      <td>输入</td>
      <td>矩阵B左右相邻元素间的内存地址偏移量（当前约束为k）。</td>
    </tr>
    <tr>
      <td>ldc（ int64_t）</td>
      <td>输入</td>
      <td>矩阵C左右相邻元素间的内存地址偏移量（当前约束为m）。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdBlasCgemm

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
      <td>transa（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵A是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>transb（asdBlasOperation_t）</td>
      <td>输入</td>
      <td>指定矩阵B是否需要转置。<ul><li>ASDBLAS_OP_N：不转置</li><li>ASDBLAS_OP_T：转置</li><li>ASDBLAS_OP_C：共轭转置</li></ul></td>
    </tr>
    <tr>
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>矩阵C的行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵C的列数。</td>
    </tr>
    <tr>
      <td>k（ int64_t）</td>
      <td>输入</td>
      <td>矩阵A和B的公共维度。</td>
    </tr>
    <tr>
      <td>lda（ int64_t）</td>
      <td>输入</td>
      <td>矩阵A左右相邻元素间的内存地址偏移量（当前约束为m）。</td>
    </tr>
    <tr>
      <td>ldb（ int64_t）</td>
      <td>输入</td>
      <td>矩阵B左右相邻元素间的内存地址偏移量（当前约束为k）。</td>
    </tr>
    <tr>
      <td>ldc（ int64_t）</td>
      <td>输入</td>
      <td>矩阵C左右相邻元素间的内存地址偏移量（当前约束为m）。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'A'</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li></ul></td>
    </tr>
    <tr>
      <td>B（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的矩阵，对应公式中的'B'</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li></ul></td>
    </tr>
    <tr>
      <td>C（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>输入/输出的矩阵，对应公式中的'C'</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[m，n]。</li></ul></td>
    </tr>
    <tr>
      <td>alpha（std::complex&ltfloat&gt *）</td>
      <td>输入</td>
      <td>对应公式中的alpha，复数标量，用于乘以矩阵乘法的结果。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

- 输入的元素个数m，n，k当前覆盖支持[1,8192]。
- 算子输入数据为列主序，输入shape为[m,k]、[k,n]、[m,n]，输出shape为[m,n]。
- 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且因使用发生了安全问题，则需用户自行承担。

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

void printTensor(std::vector<std::complex<float>> tensorData, int64_t rows, int64_t cols)
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

    int m = 3;
    int k = 3;
    int n = 3;
    asdBlasOperation_t transA = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasOperation_t transB = asdBlasOperation_t::ASDBLAS_OP_N;
    std::complex<float> alpha = std::complex<float>(1.0f, 1.0f);
    std::complex<float> beta = std::complex<float>(2.0f, 2.0f);

    int64_t lda = m;
    int64_t ldb = k;
    int64_t ldc = m;

    const int64_t tensorASize = m * k;
    const int64_t tensorBSize = k * n;
    const int64_t tensorCSize = m * n;

    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.reserve(tensorASize);
    for (int i = 0; i < tensorASize; i++) {
        tensorInAData.push_back(std::complex<float>(1.0f, i + 0.0f));
    }

    std::vector<std::complex<float>> tensorInBData;
    tensorInBData.reserve(tensorBSize);
    for (int i = 0; i < tensorBSize; i++) {
        tensorInBData.push_back(std::complex<float>(1.0f, i + 0.0f));
    }

    std::vector<std::complex<float>> tensorInCData;
    tensorInCData.reserve(tensorCSize);
    for (int i = 0; i < tensorCSize; i++) {
        tensorInCData.push_back(std::complex<float>(1.0f, i + 0.0f));
    }

    std::vector<int64_t> matAShape = {m, k};
    std::vector<int64_t> matBShape = {k, n};
    std::vector<int64_t> matCShape = {m, n};

    aclTensor *matA = nullptr;
    aclTensor *matB = nullptr;
    aclTensor *matC = nullptr;
    void *matADeviceAddr = nullptr;
    void *matBDeviceAddr = nullptr;
    void *matCDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<float>>(
        tensorInAData, matAShape, &matADeviceAddr, aclDataType::ACL_COMPLEX64, &matA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<float>>(
        tensorInBData, matBShape, &matBDeviceAddr, aclDataType::ACL_COMPLEX64, &matB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<float>>(
        tensorInCData, matCShape, &matCDeviceAddr, aclDataType::ACL_COMPLEX64, &matC);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData, m, k);
    std::cout << "------- input TensorInB -------" << std::endl;
    printTensor(tensorInBData, k, n);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCgemmPlan(handle, transA, transB, m, n, k, lda, ldb, ldc);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgemm(handle, transA, transB, m, n, k, alpha, matA, lda, matB, ldb, beta, matC, ldc));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInCData.data(),
        tensorCSize * sizeof(std::complex<float>),
        matCDeviceAddr,
        tensorCSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInC -------" << std::endl;
    printTensor(tensorInCData, m, n);

    aclDestroyTensor(matA);
    aclDestroyTensor(matB);
    aclDestroyTensor(matC);
    aclrtFree(matADeviceAddr);
    aclrtFree(matBDeviceAddr);
    aclrtFree(matCDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
