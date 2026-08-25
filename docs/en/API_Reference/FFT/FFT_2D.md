# FFT_2D

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:56:32.649Z pushedAt=2026-08-20T11:47:59.791Z -->

## Applicable Product

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √  |

## Function Description

- API function:

`asdFftMakePlan2D`: Initializes the 2D FFT configuration.\
`asdFftExecC2C`: Performs a complex-to-complex FFT.\
`asdFftExecC2R`: Performs a complex-to-real FFT.\
`asdFftExecR2C`: Performs a real-to-complex FFT.

- Formula:

The Fourier transform is a linear integral transform used to convert signals between the time domain and frequency domain, and has many applications in physics and engineering. For a signal of length N, its discrete form DFT (Discrete Fourier Transform) is expressed as follows:

  ![Formula](../figures/FFT_2D_1.png)

  By treating the coefficient matrix (N*N) and the time-domain signal (N*1) as two tensors, the DFT can be completed by directly using matrix multiplication on the NPU. However, the time complexity is too high, so a fast Fourier transform is needed. The basic principle is to use the rotational symmetry of trigonometric functions in the complex domain to split the sequence into subsequences, and then use butterfly operations to reduce the computational complexity:\
  ![Formula](../figures/FFT_ID_2.png)

## Function Prototype

```Cpp
AspbStatus asdFftMakePlan2D(
  asdFftHandle          handle, 
  int64_t               fftSizeX, 
  int64_t               fftSizeY, 
  asdFftType            fftType, 
  asdFftDirection       direction, 
  int32_t               batchSize)
```

```Cpp
AspbStatus asdFftExecC2C(
  asdFftHandle           handle, 
  const aclTensor *      input, 
  const aclTensor *      output)
```

```Cpp
AspbStatus asdFftExecC2R(
  asdFftHandle           handle, 
  const aclTensor *      input, 
  const aclTensor *      output)
```

```Cpp
AspbStatus asdFftExecR2C(
  asdFftHandle           handle, 
  const aclTensor *      input, 
  const aclTensor *      output)
```

## asdFftMakePlan2D

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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Operator handle. The <code>asdFftHandle</code> object must be created manually.</td>
    </tr>
    <tr>
      <td>fftSizeX (int64_t)</td>
      <td>Input</td>
      <td>Corresponds to "M" in the formula, indicating the FFT signal length (first dimension).</td>
    </tr>
    <tr>
      <td>fftSizeY (int64_t)</td>
      <td>Input</td>
      <td>Corresponds to "N" in the formula, indicating the FFT signal length (second dimension).</td>
    </tr>
    <tr>
      <td>fftType (asdFftType)</td>
      <td>Input</td>
      <td>FFT transform type<ul><li><code>ASCEND_FFT_C2C</code>: complex-to-complex FFT.</li><li><code>ASCEND_FFT_C2R</code>: complex-to-real FFT.</li><li><code>ASCEND_FFT_R2C</code>: real-to-complex FFT.</li></ul></td>
    </tr>
    <tr>
      <td>direction (asdFftDirection)</td>
      <td>Input</td>
      <td>Specifies a forward transform or an inverse transform.<ul><li><code>ASCEND_FFT_FORWARD</code>: forward FFT.</li><li><code>ASCEND_FFT_INVERSE</code>: inverse FFT.</li></ul></td>
    </tr>
    <tr>
      <td>batchSize (int32_t)</td>
      <td>Input</td>
      <td>Number of data batches in the FFT batch processing operation.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdFftExecC2C

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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Operator handle. The <code>asdFftHandle</code> object must be created manually.</td>
    </tr>
    <tr>
      <td>input (const aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Input shape: (batchSize, fftSizeX, fftSizeY).</li></ul></td>
    </tr>
    <tr>
      <td>output (const aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Output shape: (batchSize, fftSizeX, fftSizeY).</li></ul></td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdFftExecC2R

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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Operator handle. The <code>asdFftHandle</code> object must be created manually.</td>
    </tr>
    <tr>
      <td>input (const aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Input shape: (batchSize, fftSizeX, fftSizeY/2+1).</li></ul></td>
    </tr>
    <tr>
      <td>output (const aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Output shape: (batchSize, fftSizeX, fftSizeY).</li></ul></td>
    </tr>
    </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdFftExecR2C

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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Operator handle. The <code>asdFftHandle</code> object must be created manually.</td>
    </tr>
    <tr>
      <td>input (const aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Input shape: (batchSize, fftSizeX, fftSizeY).</li></ul></td>
    </tr>
    <tr>
      <td>output (const aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Output shape: (batchSize, fftSizeX, fftSizeY/2+1).</li></ul></td>
    </tr>
    </tbody>
    </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

`asdFftMakePlan2D`

- `fftSizeX` and `fftSizeY` must not exceed $2^{27}$, and their prime factorization must not contain any prime factor greater than 199.

- `batchSize` has no additional constraints within the allowed storage range.

- The number of input elements is theoretically supported in the range from 1 to $2^{30}$.

  - The input elements do not support `inf`, `-inf`, or `NaN`. If the input contains such values, the result is undefined.

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- **C2C_2D**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

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

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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
    // Copy data from the host side to the device side using aclrtMemcpy.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Create host-side data for the tensor.
    // int batch = 2, Nfft1 = 199 * 199, Nfft2 = 4096;
    int batch = 2, Nfft1 = 64, Nfft2 = 64; // core dd
    const int64_t inSignal = Nfft2;
    const int64_t outSignal = Nfft2;
    const int64_t tensorInSize = batch * Nfft1 * inSignal;
    const int64_t tensorOutSize = batch * Nfft1 * outSignal;
    std::vector<int64_t> selfShape = {batch, Nfft1, inSignal};
    std::vector<int64_t> outShape = {batch, Nfft1, outSignal};
    std::vector<std::complex<float>> inputHostData(tensorInSize, std::complex<float>(0, 0));
    for (int i = 0; i < tensorInSize; i++) {
        inputHostData[i] = std::complex<float>(i, i + 1);
    }
    std::vector<std::complex<float>> outHostData(tensorOutSize, std::complex<float>(0, 0));
    void *inputDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;
    ret = CreateAclTensor(inputHostData, selfShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan2D(handle, Nfft1, Nfft2, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    ASD_STATUS_CHECK(asdFftExecC2C(handle, input, out));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    asdFftDestroy(handle);

    auto size = GetShapeSize(outShape);
    std::vector<std::complex<float>> outData(size, 0);
    ret = aclrtMemcpy(outData.data(),
        outData.size() * sizeof(outData[0]),
        outDeviceAddr,
        size * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);

    // Print the first 16 values of the output tensor.
    for (int64_t i = 0; i < 16; i++) {
        std::cout << static_cast<std::complex<float>>(outData[i]) << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **C2R_2D**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

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

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Create host-side data for the tensor.
    // int batch = 32, Nfft = 128;
    // int batch = 32, Nfft = 8192;
    // int batch = 32, Nfft = 15000;

    // Create host-side data for the tensor.
    int batch = 2, Nfft1 = 128, Nfft2 = 128;
    const int64_t inSignal = Nfft2 / 2 + 1;
    const int64_t outSignal = Nfft2;
    const int64_t tensorInSize = batch * Nfft1 * inSignal;
    const int64_t tensorOutSize = batch * Nfft1 * outSignal;
    std::vector<int64_t> selfShape = {batch, Nfft1, inSignal};
    std::vector<int64_t> outShape = {batch, Nfft1, outSignal};
    std::vector<std::complex<float>> inputHostData(tensorInSize, std::complex<float>(0, 0));
    for (int i = 0; i < tensorInSize; i++) {
        inputHostData[i] = std::complex<float>(i, i + 1);
    }
    std::vector<float> outHostData(tensorOutSize, 0);
    void *inputDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;
    ret = CreateAclTensor(inputHostData, selfShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan2D(handle, Nfft1, Nfft2, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    ASD_STATUS_CHECK(asdFftExecC2R(handle, input, out));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    asdFftDestroy(handle);

    auto size = GetShapeSize(outShape);
    std::vector<float> outData(size, 0);
    ret = aclrtMemcpy(outData.data(),
        outData.size() * sizeof(outData[0]),
        outDeviceAddr,
        size * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);

    // Print the first 16 values of the output tensor.
    for (int64_t i = 0; i < 16; i++) {
        std::cout << static_cast<float>(outData[i]) << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- **R2C_2D**

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;

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

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Create the host-side data of the tensor.
    int batch = 2, Nfft1 = 1024, Nfft2 = 4096;
    const int64_t inSignal = Nfft2;
    const int64_t outSignal = Nfft2 / 2 + 1;
    const int64_t tensorInSize = batch * Nfft1 * inSignal;
    const int64_t tensorOutSize = batch * Nfft1 * outSignal;
    std::vector<int64_t> selfShape = {batch, Nfft1, inSignal};
    std::vector<int64_t> outShape = {batch, Nfft1, outSignal};
    std::vector<float> inputHostData(tensorInSize, 0);
    for (int i = 0; i < tensorInSize; i++) {
        inputHostData[i] = i;
    }
    std::vector<std::complex<float>> outHostData(tensorOutSize, std::complex<float>(0, 0));
    void *inputDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;
    ret = CreateAclTensor(inputHostData, selfShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan2D(handle, Nfft1, Nfft2, asdFftType::ASCEND_FFT_R2C, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    ASD_STATUS_CHECK(asdFftExecR2C(handle, input, out));
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    asdFftDestroy(handle);

    auto size = GetShapeSize(outShape);
    std::vector<std::complex<float>> outData(size, 0);
    ret = aclrtMemcpy(outData.data(),
        outData.size() * sizeof(outData[0]),
        outDeviceAddr,
        size * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);

    // Print the first 16 values of the output tensor.
    for (int64_t i = 0; i < 16; i++) {
        std::cout << static_cast<std::complex<float>>(outData[i]) << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
