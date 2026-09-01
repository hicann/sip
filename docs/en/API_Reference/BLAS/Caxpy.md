# Caxpy

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:48:29.931Z pushedAt=2026-08-20T11:47:59.723Z -->

## Applicable Product

|Product|Supported|
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |

## Function Description

- API function:

`asdBlasMakeCaxpyPlan`: Initializes the `Caxpy` operator configuration for the handle.
`asdBlasCaxpy`: Multiplies a complex vector by a complex scalar and then adds a complex vector.

- Formula:

  $$
  y= alpha * x + y\\ 
  $$
  Example:
The input `x` is:
`[3.0 + 4.0j, 4.0 + 4.0j]`
The input `alpha` is:
`3.0 + 4.0j`
The input `y` is:
`[3.0 + 4.0j, 4.0 + 4.0j]`
After the `asdBlasCaxpy` operator is called, the output `y` is:
`[-4.0 + 28.0j, 0.0 + 32.0j]`

## Function Prototype

```Cpp
AspbStatus asdBlasMakeCaxpyPlan(asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasCaxpy(
  asdBlasHandle              handle, 
  const int64_t              n, 
  const std::complex<float> &alpha, 
  aclTensor*                 x,
  int64_t                    incx, 
  aclTensor*                 y, 
  int64_t                    incy)
```

## asdBlasMakeCaxpyPlan

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

## asdBlasCaxpy

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
      <td>Number of complex elements in the vector.</td>
    </tr>
    <tr>
      <td>alpha (std::complex&lt;float&gt; &)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "alpha" in the formula. Input complex scalar.</li><li>Supported data type: <code>COMPLEX64</code>.</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>Input vector, corresponding to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector "x" (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Input/Output</td>
      <td><ul><li>Input/Output vector, corresponding to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector y (constrained to 1 in the current version).</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements `n` is currently supported in the range [1, 6.71e+06].

- The operator input shapes are [n] and [n], and the output shape is [n].

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

    int64_t vecSize = 8;
    int64_t resultSize = 8;
    std::complex<float> alpha = {-2.0f, -4.0f};

    std::vector<std::complex<float>> tensorInVecData;
    tensorInVecData.reserve(vecSize);
    for (int i = 0; i < vecSize; i++) {
        tensorInVecData.push_back({(float)(1.0 + i), (float)(2.0 + i)});
    }

    std::vector<std::complex<float>> tensorResultData;
    tensorResultData.reserve(resultSize);

    for (int i = 0; i < vecSize; i++) {
        tensorResultData.push_back({(float)(0.0), (float)(0.0)});
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input vec -------" << std::endl;
    printTensor(tensorInVecData, vecSize);

    std::vector<int64_t> vecShape = {vecSize};
    std::vector<int64_t> resultShape = {resultSize};

    aclTensor *inVec = nullptr;
    aclTensor *outResult = nullptr;
    void *inpVecDeviceAddr = nullptr;
    void *outResultDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<float>>(
        tensorInVecData, vecShape, &inpVecDeviceAddr, aclDataType::ACL_COMPLEX64, &inVec);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<float>>(
        tensorResultData, resultShape, &outResultDeviceAddr, aclDataType::ACL_COMPLEX64, &outResult);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCaxpyPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCaxpy(handle, vecSize, alpha, inVec, 1, outResult, 1));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorResultData.data(),
        resultSize * sizeof(std::complex<float>),
        outResultDeviceAddr,
        resultSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    printTensor(tensorResultData, resultSize);

    aclDestroyTensor(inVec);
    aclDestroyTensor(outResult);
    aclrtFree(inpVecDeviceAddr);
    aclrtFree(outResultDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
