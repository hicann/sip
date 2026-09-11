# Strmm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->

<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->

<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->

<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：\
asdBlasMakeStrmmPlan：初始化该句柄对应的Strmm算子配置。\
asdBlasStrmm：单精度算子，其功能是将一个三角矩阵A乘一个矩阵B，得到一个新的矩阵C。
- 计算公式：
  $$
  c =
  \begin{cases}
  alpha*op(A)*B & if side == ASDBLAS\_SIDE\_LEFT \\
  alpha*B*op(A) & if side == ASDBLAS\_SIDE\_RIGHT \\
  \end{cases}
  $$

  示例：\
  输入“A”为：\
  [   [ 1, 0 ],
    [ 3, 4 ]  ]\
  输入“B”为：\
 [   [ 1, 2 ],
    [ 3, 4 ]  ]\
  输入“side”为：L，“uplo”为：L，输入“trans”为：N，输入“diag”为：N。\
  输入“n”为：2，输入“lda”为：2，输入“ldb”为：2，输入“ldc”为：2。\
  输入“alpha”为：2.345。\
  调用“asdBlasStrmm”算子后，输出“C”为：\
[  [ 2.3450,  4.6900],\
   [35.1750, 51.5900]  ]

## 函数原型

```Cpp
AspbStatus asdBlasMakeStrmmPlan(
  asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasStrmm(
  asdBlasHandle          handle,
  asdBlasSideMode_t      side,
  asdBlasFillMode_t      uplo,
  asdBlasOperation_t      trans,
  asdBlasDiagType_t       diag,
  const int64_t           m,
  const int64_t           n,
  const float &           alpha,
  aclTensor *             A,
  const int64_t           lda,
  aclTensor *             B,
  const int64_t           ldb,
  aclTensor *             C,
  const int64_t           ldc)
```

## asdBlasMakeStrmmPlan

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

## asdBlasStrmm

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
      <td>side（asdBlasSideMode_t）</td>
      <td>输入</td>
      <td>指定矩阵A是乘法左侧还是右侧。<ul><li>ASDBLAS_SIDE_LEFT:左侧</li><li>ASDBLAS_SIDE_RIGHT:右侧</li></ul></td>
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
      <td>m（int64_t）</td>
      <td>输入</td>
      <td>矩阵B和C的行数。</td>
    </tr>
    <tr>
      <td>n（int64_t）</td>
      <td>输入</td>
      <td>矩阵B和C的列数。</td>
    </tr>
    <tr>
      <td>alpha（float &）</td>
      <td>输入</td>
      <td>公式中的alpha，用于计算矩阵乘法的系数。</td>
    </tr>
    <tr>
      <td>A（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>当A为乘法左矩阵时，shape为[m，m]</li><li>当A为乘法右矩阵时，shape为[n，n]</li></ul></td>
    </tr>
    <tr>
      <td>lda（int64_t）</td>
      <td>输入</td>
      <td>表示张量A中元素的间隔（当前约束为m/n，当side=ASDBLAS_SIDE_LEFT时为m，side=ASDBLAS_SIDE_RIGHT时为n）。</td>
    </tr>
    <tr>
      <td>B（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'B'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[m，n]</li></ul></td>
    </tr>
    <tr>
      <td>ldb（int64_t）</td>
      <td>输入</td>
      <td>表示张量B中元素的间隔（当前约束为m）。</td>
    </tr>
    <tr>
      <td>C（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'C'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>shape为[m，n]</li></ul></td>
    </tr>
    <tr>
      <td>ldc（int64_t）</td>
      <td>输入</td>
      <td>表示张量C中元素的间隔（当前约束为m）。</td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  - 输入的元素个数m，n当前覆盖支持[1,8193]。
  - 当side = ASDBLAS_SIDE_LEFT时，算子输入shape为[m,m]、[m,n]，输出shape为[m,n]。
  - 当side = ASDBLAS_SIDE_RIGHT时，算子输入shape为[n,n]、[m,n]，输出shape为[m，n]。
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

    asdBlasSideMode_t side = asdBlasSideMode_t::ASDBLAS_SIDE_LEFT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;
    const int64_t m = 5;
    const int64_t n = 5;
    float alpha = 1.0;
    int64_t lda = m;
    int64_t ldb = m;
    int64_t ldc = m;

    const int64_t tensorASize = m * m;
    std::vector<float> tensorInAData(tensorASize, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < m; j++) {
            tensorInAData[m * i + j] = i;
        }
    }

    const int64_t tensorBSize = m * n;
    std::vector<float> tensorInBData(tensorBSize, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInBData[n * i + j] = i;
        }
    }

    const int64_t tensorCSize = m * n;
    std::vector<float> tensorCData(tensorCSize, 0.0);

    std::cout << "side = " << static_cast<int32_t>(side) << std::endl;
    std::cout << "uplo = " << static_cast<int32_t>(uplo) << std::endl;
    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "diag = " << static_cast<int32_t>(diag) << std::endl;

    std::cout << "------- input A -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < m; j++)
            std::cout << tensorInAData[i * m + j] << " ";
        std::cout << std::endl;
    }

    std::cout << "------- input B -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++)
            std::cout << tensorInBData[i * n + j] << " ";
        std::cout << std::endl;
    }

    std::vector<int64_t> aShape = {tensorASize};
    std::vector<int64_t> bShape = {tensorBSize};
    std::vector<int64_t> cShape = {tensorCSize};

    aclTensor *inputA = nullptr;
    aclTensor *inputB = nullptr;
    aclTensor *outputC = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputBDeviceAddr = nullptr;
    void *outputCDeviceAddr = nullptr;

    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_FLOAT, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInBData, bShape, &inputBDeviceAddr, aclDataType::ACL_FLOAT, &inputB);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorCData, cShape, &outputCDeviceAddr, aclDataType::ACL_FLOAT, &outputC);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, inputA, lda, inputB, ldb, outputC, ldc));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorCData.data(),
        tensorCSize * sizeof(float),
        outputCDeviceAddr,
        tensorCSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output C -------" << std::endl;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            std::cout << tensorCData[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(outputC);
    aclrtFree(inputADeviceAddr);
    aclrtFree(inputBDeviceAddr);
    aclrtFree(outputCDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
