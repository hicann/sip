# Colwise_mul

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:49:37.018Z pushedAt=2026-08-20T11:47:59.733Z -->

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

`asdBlasMakeColwiseMulPlan`: initializes the ColwiseMul operator configuration for the handle.\
`asdBlasColwiseMul`: performs column-wise element-wise multiplication of a complex matrix and a complex vector, and returns a complex matrix with the same shape and size as the input matrix.

- Formula:

 ![Formula](../figures/colwise.png)
  Example:\
The input `A` is:\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
The input `X` is:\
[ 1+1i, 2+2i ]\
After the `asdBlasColwiseMul` operator is called, the output `result` is:\
[ [ 0+2i, 0+2i ],\
  [ 0+8i, 0+8i ] ]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeColwiseMulPlan(
  asdBlasHandle       handle)

```

```Cpp
AspbStatus asdBlasColwiseMul(
  asdBlasHandle       handle, 
  const int64_t       m, 
  const int64_t       n, 
  aclTensor *         mat, 
  aclTensor *         vec,
  aclTensor *         result)
```

## asdBlasMakeColwiseMulPlan

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

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasColwiseMul

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
      <td>mat (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>Input vector, corresponding to "A" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m, n].</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix mat, and number of elements of vector vec.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of the matrix mat.</td>
    </tr>
    <tr>
      <td>vec (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>Input vector, corresponding to "X" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m].</li></ul></td>
    </tr>
    <tr>
      <td>result (aclTensor*)</td>
      <td>Output</td>
      <td><ul><li>Output vector, corresponding to "result" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m, n].</li></ul></td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

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
    // Call aclrtMemcpy to copy data from the host side to the device memory.
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
    int64_t n = 2;
    int64_t matSize = m * n;

    std::vector<std::complex<float>> tensorInMatData;
    tensorInMatData.reserve(matSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInMatData[n * i + j] = (std::complex<float>){2.0, -2.0};
        }
    }

    int64_t vecSize = m;
    std::vector<std::complex<float>> tensorInVecData;
    tensorInVecData.reserve(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInVecData[i] = (std::complex<float>){3.0, -4.0};
    }

    int64_t resultSize = m * n;
    std::vector<std::complex<float>> resultData;
    resultData.reserve(resultSize);

    std::cout << "------- input mat -------" << std::endl;
    printTensor(tensorInMatData.data(), m, n);

    std::cout << "------- input vec -------" << std::endl;
    printTensor(tensorInVecData.data(), m, 1);

    std::vector<int64_t> matShape = {m, n};
    std::vector<int64_t> vecShape = {m};
    std::vector<int64_t> resultShape = {m, n};

    aclTensor *inputMat = nullptr;
    aclTensor *inputVec = nullptr;
    aclTensor *outputResult = nullptr;
    void *inputMatDeviceAddr = nullptr;
    void *inputVecDeviceAddr = nullptr;
    void *outputResultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInMatData, matShape, &inputMatDeviceAddr, aclDataType::ACL_COMPLEX64, &inputMat);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInVecData, vecShape, &inputVecDeviceAddr, aclDataType::ACL_COMPLEX64, &inputVec);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &outputResultDeviceAddr, aclDataType::ACL_COMPLEX64, &outputResult);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeColwiseMulPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasColwiseMul(handle, m, n, inputMat, inputVec, outputResult));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);
    buffer = nullptr;

    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(std::complex<float>),
        outputResultDeviceAddr,
        resultSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    printTensor(resultData.data(), m, n);

    aclDestroyTensor(inputMat);
    aclDestroyTensor(inputVec);
    aclDestroyTensor(outputResult);
    aclrtFree(inputMatDeviceAddr);
    aclrtFree(inputVecDeviceAddr);
    aclrtFree(outputResultDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
