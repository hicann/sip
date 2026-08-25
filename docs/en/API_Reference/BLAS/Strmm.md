# Strmm

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:53:27.367Z pushedAt=2026-08-20T11:47:59.769Z -->

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |

## Function Description

- API function:

`asdBlasMakeStrmmPlan`: initializes the `Strmm` operator configuration corresponding to the handle.\
`asdBlasStrmm`: a single-precision operator that multiplies a triangular matrix A by a matrix B to obtain a new matrix C.

- Formula:

  $$
  c = 
  \begin{cases}
  alpha*op(A)*B & if side == ASDBLAS\_SIDE\_LEFT \\
  alpha*B*op(A) & if side == ASDBLAS\_SIDE\_RIGHT \\
  \end{cases}
  $$

  Example:\
  The input `A` is:\
  [   [ 1, 0 ],
    [ 3, 4 ]  ]\
  The input `B` is:\
 [   [ 1, 2 ],
    [ 3, 4 ]  ]\
  The input `side` is L, `uplo` is L, `trans` is N, and `diag` is N.\
  The input `n` is 2, `lda` is 2, `ldb` is 2, and `ldc` is 2.\
  The input `alpha` is 2.345.\
  After the `asdBlasStrmm` operator is called, the output `C` is:\
[  [ 2.3450,  4.6900],\
   [35.1750, 51.5900]  ]

## Function Prototype

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

- **Parameter description:**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Operator handle</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  Returns status codes. For details, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasStrmm

- **Parameter description:**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the operator.</td>
    </tr>
    <tr>
      <td>side (asdBlasSideMode_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A is on the left or right side of the multiplication.<ul><li><code>ASDBLAS_SIDE_LEFT</code>: left side</li><li><code>ASDBLAS_SIDE_RIGHT</code>: right side</li></ul></td>
    </tr>
    <tr>
      <td>uplo (asdBlasFillMode_t)</td>
      <td>Input</td>
      <td>Specifies the storage format of matrix A.<ul><li><code>ASDBLAS_FILL_MODE_LOWER</code>: lower triangular</li><li><code>ASDBLAS_FILL_MODE_UPPER</code>: upper triangular</li></ul></td>
    </tr>
    <tr>
      <td>diag (asdBlasDiagType_t)</td>
      <td>Input</td>
      <td>Specifies whether the diagonal elements of matrix A are assumed to be 1.<ul><li><code>ASDBLAS_DIAG_NON_UNIT</code>: not assumed to be 1</li><li><code>ASDBLAS_DIAG_UNIT</code>: assumed to be 1</li></ul></td>
    </tr>
    <tr>
      <td>trans (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A is transposed.<ul><li><code>ASDBLAS_OP_N</code>: not transposed</li><li><code>ASDBLAS_OP_T</code>: transposed</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrices B and C.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrices B and C.</td>
    </tr>
    <tr>
      <td>alpha (float &)</td>
      <td>Input</td>
      <td>Alpha in the formula, used as the coefficient for matrix multiplication.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "A" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>When A is the left matrix in the multiplication, the shape is [m, m]</li><li>When A is the right matrix in the multiplication, the shape is [n, n]</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Element stride in tensor A (currently constrained to m/n: m is used when side = <code>ASDBLAS_SIDE_LEFT</code> and n is used when side = <code>ASDBLAS_SIDE_RIGHT</code>).</td>
    </tr>
    <tr>
      <td>B (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "B" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [m, n]</li></ul></td>
    </tr>
    <tr>
      <td>ldb (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stride between elements in tensor B (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>C (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Corresponds to "C" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [m, n]</li></ul></td>
    </tr>
    <tr>
      <td>ldc (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stride of elements in tensor C (currently constrained to <code>m</code>).</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

  - The number of input elements `m` and `n` is currently supported in the range [1, 8193].

  - When side = ASDBLAS_SIDE_LEFT, the operator input shapes are [m, m] and [m, n], and the output shape is [m, n].

  - When side = ASDBLAS_SIDE_RIGHT, the operator input shapes are [n, n] and [m, n], and the output shape is [m, n].

  - During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

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
    // Boilerplate: Initialize ACL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute strides for a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
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
