# CmatinvBatched

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:49:49.878Z pushedAt=2026-08-20T11:47:59.736Z -->

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

`asdBlasMakeCmatinvBatchedPlan`: initializes the operator configuration corresponding to this handle.\
`asdBlasCmatinvBatched`: computes the inverse of complex matrices.

- Formula:

Computes the inverse matrix of a batch of complex matrices. Each complex matrix must satisfy $A^{-1}A=I$, where `A` is a non-singular square matrix of size n×n and `I` is the identity matrix.

  Example:\
The input `A` is:\
[2-2i ,1-i ,1-i, 1-i\
1-i, 2-2i, 1-i ,1-i\
1-i, 1-i ,2-2i ,1-i\
1-i ,1-i, 1-i, 2-2i]\

  [3-3i ,1-i, 1-i, 1-i\
1-i, 3-3i, 1-i ,1-i\
1-i, 1-i ,3-3i ,1-i\
1-i, 1-i, 1-i, 3-3i]\
The input `n` is: 4\
The input `batchSize` is: 2\
After the `asdBlasCmatinvBatched` operator is called,\
the output `Ainv` is:\
[0.4+0.4i, -0.1-0.1i,-0.1-0.1i ,-0.1-0.1i\
-0.1-0.1i ,0.4+0.4i ,-0.1-0.1i ,-0.1-0.1i\
-0.1-0.1i ,-0.1-0.1i,0.4+0.4i ,-0.1-0.1i\
-0.1-0.1i ,-0.1-0.1i,-0.1-0.1i,0.4+0.4i]

   [0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i\
-0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i\
-0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i\
-0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeCmatinvBatchedPlan(
  asdBlasHandle            handle, 
  const int64_t            n,
   const int64_t           batchSize)
```

```Cpp
AspbStatus asdBlasCmatinvBatched(
  asdBlasHandle            handle, 
  const int64_t            n, 
  aclTensor *              A, 
  const int64_t            lda,
  aclTensor *              Ainv, 
  const int64_t            lda_inv, 
  aclTensor *              info, 
  int64_t                  batchSize)
```

## asdBlasMakeCmatinvBatchedPlan

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
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Number of matrices involved in complex matrix inversion.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasCmatinvBatched

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
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "A" in the formula.</li><li>Row-major order.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batchCount, n, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements in the same row of A (constrained to n in the current version).</td>
    </tr>
    <tr>
      <td>Ainv (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Output inverse matrix.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, n, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda_inv (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements in the left and right directions of the output inverse matrix (constrained to n in the current version).</td>
    </tr>
    <tr>
      <td>info (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Inversion result information of each batch matrix.</li><li>Supported data type: <code>int32_t</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, 1].</li></ul></td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Number of matrices involved in complex matrix inversion.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The `lda`, `lda_inv`, and `info` parameters are not actually enabled in the current version.

- The input parameter `n` is less than or equal to 256.

- The input parameter `batchSize` is less than or equal to 3000.

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

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

void printTensor(const std::complex<float> *tensorData, int64_t batchSize, int64_t rows, int64_t cols)
{
    for(int64_t b = 0; b < batchSize; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                std::cout << tensorData[b * rows * cols + i * cols + j] << " ";
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

    // Compute the strides of a contiguous tensor.
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

    int64_t batchSize = 3;
    int64_t n = 4;

    int64_t tensorASize = batchSize * n * n;
    std::vector<std::complex<float>> tensorInAData;
    std::vector<std::complex<float>> tensorInAinvData;
    std::vector<int32_t> tensorInInfoData;
    tensorInAData.reserve(tensorASize);
    tensorInAinvData.reserve(tensorASize);
    tensorInInfoData.reserve(batchSize);

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                if (i == j) {
                    tensorInAData[n * n * batchIdx + n * i + j] = {2.0f + batchIdx, -2.0f - batchIdx};
                } else {
                    tensorInAData[n * n * batchIdx + n * i + j] = {1.0f, -1.0f};
                }
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                tensorInAinvData[n * n * batchIdx + n * i + j] = {-1.0f, -1.0f};
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
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInAinvData, ainvShape, &inputAinvDeviceAddr, aclDataType::ACL_COMPLEX64, &inputAinv);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInInfoData, infoShape, &inputInfoDeviceAddr, aclDataType::ACL_INT32, &inputInfo);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCmatinvBatchedPlan(handle, n, batchSize);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);
    asdBlasSynchronize(handle);

    ASD_STATUS_CHECK(asdBlasCmatinvBatched(handle, n, inputA, n, inputAinv, n, inputInfo, batchSize));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAinvData.data(),
        tensorASize * sizeof(std::complex<float>),
        inputAinvDeviceAddr,
        tensorASize * sizeof(std::complex<float>),
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
