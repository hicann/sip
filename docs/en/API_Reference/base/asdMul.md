# asdMul

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:46:31.004Z pushedAt=2026-08-20T11:47:59.697Z -->

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |

## Function Description

- API function: Supports element-wise vector product (Hadamard product), and returns a complex matrix with the same shape and size as the input.

- Formula:

  $$
  result=A \odot\ B =(A)_{ij}(B)_{ij}
  $$

    Example:

  The input `A` is:\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
The input `B` is:\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
After the `asdMul` operator is called, the output `result` is:\
[ [ 0+2i, 0+2i ],\
  [ 0+8i, 0+8i ] ]

## Function Prototype

```Cpp
AspbStatus asdMul(
  int            n, 
  const void *   x, 
  const void *   y, 
  const void *   z, 
  void *         stream, 
  void *         workspace = nullptr)
```

## asdMul

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
      <td>n (int)</td>
      <td>Input</td>
      <td>Represents the number of input elements.</td>
    </tr>
    <tr>
      <td>x (void *)</td>
      <td>Input</td>
      <td><ul><li>Represents the input matrix, corresponding to "A" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>y (void *)</td>
      <td>Input</td>
      <td><ul><li>Represents the input matrix, corresponding to "B" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>z (void *)</td>
      <td>Output</td>
      <td><ul><li>Represents the output matrix, corresponding to "result" in the formula.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>NPU execution stream.</td>
    </tr>
    <tr>
      <td>workspace (void *)</td>
      <td>Input</td>
      <td>Workspace required by the <code>asdMul</code> operator.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

  - The number of input elements `n` is theoretically supported in the range [1, 9.22e+18].

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- **mul_complex32**

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
        }                                                \
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

void printTensor(const std::complex<op::fp16_t> *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
        std::cout << "(" << (float)tensorData[i].real() << "," << (float)tensorData[i].imag() << ")" << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t n = 8;

    int64_t vecSize = n;
    std::vector<std::complex<op::fp16_t>> tensorInXData;
    std::vector<std::complex<op::fp16_t>> tensorInYData;
    tensorInXData.reserve(vecSize);
    tensorInYData.reserve(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInXData.push_back({(op::fp16_t)(9.0f + i), (op::fp16_t)(100.0f + i)});
    }
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInYData.push_back({(op::fp16_t)(22.0f + i), (op::fp16_t)(33.0f * (i + 1))});
    }
    std::vector<std::complex<op::fp16_t>> tensorOutZData(
        vecSize, {(op::fp16_t)0.0f, (op::fp16_t)0.0f});

    std::cout << "------- input X -------" << std::endl;
    printTensor(tensorInXData.data(), vecSize);
    std::cout << "------- input Y -------" << std::endl;
    printTensor(tensorInYData.data(), vecSize);

    std::vector<int64_t> xShape = {vecSize};
    std::vector<int64_t> yShape = {vecSize};
    std::vector<int64_t> zShape = {vecSize};

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX32, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX32, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutZData, zShape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX32, &outputZ);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ASD_STATUS_CHECK(asdMul(n, inputX, inputY, outputZ, stream));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutZData.data(),
        vecSize * sizeof(std::complex<op::fp16_t>),
        outputZDeviceAddr,
        vecSize * sizeof(std::complex<op::fp16_t>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy z from device to host failed. ERROR: %d\n", ret); return ret);
    std::cout << "------- output Z -------" << std::endl;

    printTensor(tensorOutZData.data(), vecSize);
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(outputZ);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(outputZDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **mul_complex64**

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                            \
    do {                                                 \
        AsdSip::AspbStatus err_ = (err);                 \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {    \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                    \
        }                                                \
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

void printTensor(const std::complex<float> *tensorData, int64_t nums)
{
    for (int64_t i = 0; i < nums; i++) {
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

    int64_t n = 8;

    int64_t vecSize = n;
    std::vector<std::complex<float>> tensorInXData;
    std::vector<std::complex<float>> tensorInYData;
    tensorInXData.reserve(vecSize);
    tensorInYData.reserve(vecSize);
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInXData[i] = {(float)(1.0 + i), (float)(1.0 + i)};
    }
    for (int64_t i = 0; i < vecSize; i++) {
        tensorInYData[i] = {(float)(2.0 + i), 3.0};
    }
    std::vector<std::complex<float>> tensorOutZData(vecSize, {0.0f, 0.0f});

    std::cout << "------- input X -------" << std::endl;
    printTensor(tensorInXData.data(), vecSize);
    std::cout << "------- input Y -------" << std::endl;
    printTensor(tensorInYData.data(), vecSize);

    std::vector<int64_t> xShape = {vecSize};
    std::vector<int64_t> yShape = {vecSize};
    std::vector<int64_t> zShape = {vecSize};

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutZData, zShape, &outputZDeviceAddr, aclDataType::ACL_COMPLEX64, &outputZ);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ASD_STATUS_CHECK(asdMul(n, inputX, inputY, outputZ, stream));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutZData.data(),
        vecSize * sizeof(std::complex<float>),
        outputZDeviceAddr,
        vecSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy z from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- Output -------" << std::endl;
    printTensor(tensorOutZData.data(), vecSize);
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(outputZ);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(outputZDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
