# Csrot

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:50:44.947Z pushedAt=2026-08-20T11:47:59.745Z -->

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

- API function:\

`asdBlasMakeRotPlan`: Initializes the Csrot operator configuration corresponding to this handle.\
`asdBlasCsrot`: Rotates the input complex vector group (x, y).

- Formula:

  $$
  \begin{bmatrix}
  x\\y\end{bmatrix}=\begin{bmatrix}
  c & s\\-s & c\end{bmatrix}*\begin{bmatrix}
  x\\y\end{bmatrix}
  $$
  where x[i] = c\*x[i] + s\*y[i], y[i] = -s\*x[i] + c\*y[i]\
  c is the cosine of the rotation angle, s is the sine of the rotation angle, and x and y are complex vectors.\
  Example:
The input `x` is:\
[3.0 + 4.0j, 2.0 + 1.0j]\
The input `y` is:\
[1.0 + 1.0j, 3.0 + 3.0j]\
The input `c` is:\
$\frac{\sqrt{3}}{2}$\
The input `s` is:\
0.5\
After the `asdBlasCsrot` operator is called, the output `x` is:\
[3.098076+3.9641016j, 3.232051+2.3660254j]\
The output `y` is:\
[-0.6339746-1.1339746j, 1.5980761+2.098076j]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeRotPlan(
  asdBlasHandle handle)
```

```Cpp
AspbStatus asdBlasCsrot(
  asdBlasHandle     handle, 
  const int64_t     n, 
  aclTensor *       x, 
  const int64_t     incx, 
  aclTensor *       y,
  const int64_t     incy, 
  const float &     c, 
  const float &     s)
```

## asdBlasMakeRotPlan

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
  </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasCsrot

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
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of complex elements in the vector.</td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li>
      <li>Shape: [n]</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of x (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>y (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li></ul>
      <li>Shape: [n]</li></td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of y (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>c (float &)</td>
      <td>Input</td>
      <td>Pointer to the cosine value of the rotation matrix.</td>
    </tr>
    <tr>
      <td>s (float &)</td>
      <td>Input</td>
      <td>Pointer to the sine value of the rotation matrix.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements `n` is currently supported in the range [1, 2.50e+06].

- The operator input shape is [n], and the output shape is [n].

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- **asdBlasCsrot**

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"
#include "utils/mem_base.h"

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
    // Copy data from the host side to the device side memory by calling `aclrtMemcpy`.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create an `aclTensor` by calling the `aclCreateTensor` API.
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

void printTensor(const std::complex<float> *tensorData, int64_t tensorSize)
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
    int incx = 1;
    int incy = 1;
    const float cosValue = sqrt(3) / 2;
    const float sinValue = 0.5;

    int64_t xSize = n;
    int64_t ySize = n;

    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < n; i++) {
        tensorInXData[i] = (std::complex<float>){2.0, 3.0};
    }

    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.reserve(ySize);
    for (int64_t i = 0; i < n; i++) {
        tensorInYData[i] = (std::complex<float>){5.0, 6.0};
    }

    std::cout << "cosValue = " << cosValue << std::endl;
    std::cout << "sinValue = " << sinValue << std::endl;
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), ySize);

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeRotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCsrot(handle, n, inputX, incx, inputY, incy, cosValue, sinValue));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInXData.data(),
        xSize * sizeof(std::complex<float>),
        inputXDeviceAddr,
        xSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor x from device to host failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorInYData.data(),
        ySize * sizeof(std::complex<float>),
        inputYDeviceAddr,
        ySize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor y from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);
    std::cout << "------- output TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), ySize);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
