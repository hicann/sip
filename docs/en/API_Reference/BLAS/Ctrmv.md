# Ctrmv

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:51:22.440Z pushedAt=2026-08-20T11:47:59.751Z -->

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

  `asdBlasMakeCtrmvPlan`: Initializes the `Ctrmv` operator configuration corresponding to this handle.\
  `asdBlasCtrmv`: A single-precision complex matrix-vector multiplication operator, used to compute the product of a complex triangular matrix and a complex vector.

- Formula:

  - Matrix-vector product

  $$
  x = A * x
  $$

  - Transpose matrix-vector product

  $$
  x = A^T * x
  $$

  - Conjugate transpose matrix-vector product

  $$
  x = A^H * x
  $$
  Example:\
  The input `A` is:\
  [[ 1+2i, 1+2i ],\
  [ 1+2i, 1+2i ]]\
  The input `X` is:\
  [  [ 0+0i, 1+i ]  ]\
  The input `uplo` is L, the input `trans` is N, and the input `diag` is N.\
  The input `n` is 2, the input `lda` is 2, and the input `incx` is 1.\
  After the `asdBlasCtrmv` operator is called, the output "x" is:\
  [  [ 0+0i, -1+3i ]  ]

## Function Prototype

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
      <td>uplo (asdBlasFillMode_t)</td>
      <td>Input</td>
      <td>Specifies the storage format of matrix A.<ul><li><code>ASDBLAS_FILL_MODE_LOWER</code>: lower triangular</li><li><code>ASDBLAS_FILL_MODE_UPPER</code>: upper triangular</li></ul></td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Order of matrix A, and dimension of vector x.</td>
    </tr>
  </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasCtrmv

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
    <tr>
      <td>uplo (asdBlasFillMode_t)</td>
      <td>Input</td>
      <td>Storage format of the specified matrix A.<ul><li><code>ASDBLAS_FILL_MODE_LOWER</code>: lower triangular</li><li><code>ASDBLAS_FILL_MODE_UPPER</code>: upper triangular</li></ul></td>
    </tr>
    <tr>
      <td>trans (asdBlasDiagType_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>diag (asdBlasDiagType_t)</td>
      <td>Input</td>
      <td>Handling method for the diagonal elements of the specified matrix A.<ul><li><code>ASDBLAS_DIAG_NON_UNIT</code>: general matrix</li><li><code>ASDBLAS_DIAG_UNIT</code>: identity matrix</li></ul></td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Order of matrix A, dimension of vector x.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "A" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n, n]</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Size of the first dimension of matrix A.</td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Stride of elements in vector x.</td>
    </tr>
  </tbody>
    </table>

- **Return value:**

  Return values. For details, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements `n` is currently supported in the range [1, 8192].

- The operator input shape is [n, n], [n], and the output shape is [n].

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

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
    // Copy host-side data to device-side memory using aclrtMemcpy.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
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
    tensorInXData.reserve(tensorXSize);
    for (int64_t i = 0; i < tensorXSize; i++) {
        tensorInXData[i] = {(float)(1.0 * i), (float)(1.0 * i)};
    }

    int64_t tensorASize = n * n;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.reserve(tensorASize);
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
