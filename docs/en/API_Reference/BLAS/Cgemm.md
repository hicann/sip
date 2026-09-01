# Cgemm

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:48:31.239Z pushedAt=2026-08-20T11:47:59.726Z -->

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

`asdBlasMakeCgemmPlan`: initializes the Cgemm operator configuration corresponding to the handle.\
`asdBlasCgemm`: a type of matrix multiplication operation, used to compute the product of two complex matrices.

- Formula:

  $$
  C= alpha * op(A)*op(B) + beta * C\\
  \text{where } op(X)= X \ \ \  \text{or} \ \ \  op(X) = X^T \ \ \  \text{or} \ \ \   op(X) = X^H 
  $$
  Example:\
The input `inTensorA` is:\
[   [ 1+i, 1+2i ],
    [ 1+3i, 1+4i ]  ]\
The input `inTensorB` is:\
[   [ 2+i, 2+2i ],
    [ 2+3i, 2+4i ]  ]\
The input `inTensorC` is:\
[   [ 3+i, 3+2i ],
    [ 3+3i, 3+4i ]  ]\
The input `transa` is: N, and the input `transb` is: T.\
The input `m` is: 2, the input `n` is: 2, the input `k` is: 2, the input `alpha` is: 1+i, and `beta` is: 2+2i.\
The input `lda` is: 2, the input `ldb` is: 2, and the input `ldc` is: 2.\
After the `Cgemm` operator is called, the output `outTensor` is:\
[   [ -6+16i, -18+16i ],
    [ -24+22i, -44+14i ]  ]

## Function Prototype

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
      <td>Handle of the operator</td>
    </tr>
    <tr>
      <td>transa (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: No transpose</li><li><code>ASDBLAS_OP_T</code>: Transpose</li><li><code>ASDBLAS_OP_C</code>: Conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>transb (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix B needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: No transpose</li><li><code>ASDBLAS_OP_T</code>: Transpose</li><li><code>ASDBLAS_OP_C</code>: Conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrices A and C.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrices B and C.</td>
    </tr>
    <tr>
      <td>k (int64_t)</td>
      <td>Input</td>
      <td>Common dimension of matrices A and B.</td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix A (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>ldb (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix B (constrained to k in the current version).</td>
    </tr>
    <tr>
      <td>ldc (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix C (constrained to m in the current version).</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasCgemm

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
      <td>transa (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: No transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: Conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>transb (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix B needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: No transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: Conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix C.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrix C.</td>
    </tr>
    <tr>
      <td>k (int64_t)</td>
      <td>Input</td>
      <td>Common dimension of matrices A and B.</td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix A (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>ldb (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix B (constrained to k in the current version).</td>
    </tr>
    <tr>
      <td>ldc (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between left and right adjacent elements of matrix C (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "A" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li></ul></td>
    </tr>
    <tr>
      <td>B (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "B" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li></ul></td>
    </tr>
    <tr>
      <td>C (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Input/output matrix, corresponding to "C" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m, n].</li></ul></td>
    </tr>
    <tr>
      <td>alpha (std::complex&lt;float&gt; *)</td>
      <td>Input</td>
      <td>Corresponding to "alpha" in the formula; a complex scalar used to multiply the result of the matrix multiplication.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements `m`, `n`, and `k` is currently supported in the range [1, 8192].

- The operator input data is in column-major order. The input shapes are [m, k], [k, n], and [m, n], and the output shape is [m, n].

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator with the most concise code, rather than providing production-level security assurance. Users are advised not to directly use the example code as business code. If users use the example code in their own real business scenarios and security issues arise as a result, the users shall bear the consequences themselves.

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
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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
