# Strmv

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:54:18.184Z pushedAt=2026-08-20T11:47:59.779Z -->

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

`asdBlasMakeStrmvPlan`: Initializes the `Strmv` operator configuration for the corresponding handle.\
`asdBlasStrmv`: Single-precision API, used to compute the matrix-vector multiplication of a triangular matrix and a vector.

- Formula:

Matrix-vector product:
  $$
  x = A * x
  $$

  Transpose matrix-vector product:
  $$
  x = A^T * x
  $$

  Example:\
  The input `A` is:\
 [   [ 1, 2 ], 
    [ 3, 4 ]  ]\
  The input `x` is:\
 [1,2]\
  The input `uplo` is U, the input `trans` is T, and the input `diag` is N.\
The input `n` is 2, the input `lda` is 2, and the input `incx` is 1.\
After the `asdBlasStrmv` operator is called, the output `x` is:\
[7,10]

## Function Prototype

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
      <td>Operator handle.</td>
    </tr>
    <tr>
      <td>uplo (asdBlasFillMode_t)</td>
      <td>Input</td>
      <td>Specifies the storage format of matrix A.<ul><li><code>ASDBLAS_FILL_MODE_LOWER</code>: lower triangular</li><li><code>ASDBLAS_FILL_MODE_UPPER</code>: upper triangular</li></ul></td>
    </tr>
    <tr>
      <td>trans (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether to transpose matrix A.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li></ul></td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows and columns of matrix A, and number of elements of vector x.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasStrmv

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
      <td>Operator handle.</td>
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
      <td>Specifies whether to transpose matrix A.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li></ul></td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows and columns of matrix A, and number of elements in vector x.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "A" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n, n]</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to <code>x</code> in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stride between elements in tensor A.</td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Indicates the element stride in vector x.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  Return status codes. For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).    

## Constraints

  - The number of input elements `n` is currently supported in the range [1, 8192].

  - The input shapes of the operator are [n, n] and [n], and the output shape is [n].

  - During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

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
    // Copy the host-side data to the device-side memory by calling aclrtMemcpy.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create an aclTensor by calling the aclCreateTensor API.
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
    tensorInAData.reserve(tensorASize);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i + 1; j++) {
            tensorInAData[n * i + j] = 1.0 + i * n + j;
        }
        for (int64_t j = i + 1; j < n; j++) {
            tensorInAData[n * i + j] = 0.0;
        }
    }

    std::vector<float> tensorInXData;
    tensorInXData.reserve(tensorXSize);
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
