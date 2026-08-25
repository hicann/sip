# asdConvolve

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-19T08:17:33.442Z pushedAt=2026-08-20T11:51:29.523Z -->

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×  |

## Function Description

- API function: performs a one-dimensional filtering operation on the given signal.

- Formula:

  $$
  w(k)=\sum _{j}u(j)v(k-j+1)
  $$
  where `w(k)` is the element at position k of the output, `u(j)` is the one-dimensional signal at input position `j`, and `v(k-j+1)` is the filter convolution kernel at position `k-j+1`. The one-dimensional signal is a complex vector, and the filter convolution kernel is a real vector.

  Example:\
  The input `u` is:\
  [[1.+1.j 2.+2.j]
  [1.+1.j 2.+2.j]]
  The input `v` is:\
  [1. 2. 3. 4.]\
  After the `asdConvolve` operator is called, the output `result` is:\
  [[4.+4.j, 7.+7.j],
  [4.+4.j, 7.+7.j]]

## Function Prototype

```Cpp
AspbStatus asdConvolve(
  const aclTensor *    signal, 
  const aclTensor *    kernel, 
  aclTensor *          output, 
  asdConvolveMode_t    mode, 
  void *stream, void * workspace)
```

## asdConvolve

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
      <td>signal (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input one-dimensional signal.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>Input signal shape: [BatchCount, n].</li></ul></td>
    </tr>
    <tr>
      <td>kernel (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input filter convolution kernel.</li><li>Supported data types: <code>FLOAT16</code> and <code>FLOAT32</code>.</li><li>Input filter convolution kernel shape: [k].</li></ul></td>
    </tr>
    <tr>
      <td>output (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Input/output signal.</li><li>Supported data types: <code>COMPLEX32</code> and <code>COMPLEX64</code>.</li><li>The output shape remains consistent with the input shape.</li></ul></td>
    </tr>
    <tr>
      <td>mode (asdConvolveMode_t)</td>
      <td>Input</td>
      <td>Filter convolution mode. Currently only <code>ASD_CONVOLVE_SAME</code> is supported, which keeps the input and output vector dimensions consistent.</td>
    </tr>
    <tr>
      <td>stream (void*)</td>
      <td>Input</td>
      <td>Stream used during operator execution.</td>
    </tr>
    <tr>
      <td>workspace (void*)</td>
      <td>Input</td>
      <td>Pointer to the workspace required by the operator.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

None

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

```Cpp
#include <iostream>
#include "asdsip.h"
#include "filter_api.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl_meta.h"
#include <complex>
#include <vector>

using namespace AsdSip;

using half = op::fp16_t;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ACL_SUCCESS) {                                      \
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
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create the aclTensor.
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

template <typename T>
void printTensor(std::vector<T> tensorData, int64_t tensorSize)
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

    int64_t signalLen = 128; // 26208
    int64_t kernelLen = 32;
    int64_t batchCount = 2; // 768

    std::vector<std::complex<half>> tensorSignalData;
    tensorSignalData.reserve(signalLen * batchCount);

    std::vector<half> tensorKernelData;
    tensorKernelData.reserve(kernelLen);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorSignalData[i] = {(half)1.0, (half)1.0};
    }

    for (int64_t i = 0; i < kernelLen; i++) {
        tensorKernelData[i] = (half)(1.0 + i);
        // tensorKernelData[i] = 1.0;
    }

    std::vector<std::complex<half>> tensorOutData;
    tensorOutData.reserve(signalLen * batchCount);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorOutData[i] = {(half)-1.0, (half)-1.0};
    }

    std::vector<int64_t> signalShape = {batchCount, signalLen};
    std::vector<int64_t> kernelShape = {kernelLen};
    std::vector<int64_t> resultShape = {batchCount, signalLen};

    aclTensor *signal = nullptr;
    aclTensor *kernel = nullptr;
    aclTensor *output = nullptr;
    void *signalDeviceAddr = nullptr;
    void *kernelDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<half>>(
        tensorSignalData, signalShape, &signalDeviceAddr, aclDataType::ACL_COMPLEX32, &signal);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<half>(
        tensorKernelData, kernelShape, &kernelDeviceAddr, aclDataType::ACL_FLOAT16, &kernel);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<half>>(
        tensorOutData, resultShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX32, &output);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t lwork = 0;
    AsdSip::asdConvolveGetWorkspaceSize(signalLen, kernelLen, lwork);
    void *buffer = nullptr;

    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(AsdSip::asdConvolve(signal, kernel, output, asdConvolveMode_t::ASD_CONVOLVE_SAME, stream, buffer));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutData.data(),
        signalLen * batchCount * sizeof(std::complex<half>),
        outputDeviceAddr,
        signalLen * batchCount * sizeof(std::complex<half>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int batchIdx = 0; batchIdx < batchCount; batchIdx++) {
        for (int i = 0; i < signalLen; i++) {
            std::cout << "(" << (float)tensorOutData[batchIdx * signalLen + i].real() << ","
                      << (float)tensorOutData[batchIdx * signalLen + i].imag() << ")"
                      << " ";
        }
        std::cout << std::endl;
    }

    aclDestroyTensor(signal);
    aclDestroyTensor(kernel);
    aclDestroyTensor(output);
    aclrtFree(signalDeviceAddr);
    aclrtFree(kernelDeviceAddr);
    aclrtFree(outputDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
