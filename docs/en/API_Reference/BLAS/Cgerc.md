# Cgerc

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:50:11.033Z pushedAt=2026-08-20T11:47:59.740Z -->

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

`asdBlasMakeCgercPlan`: initializes the `Cgerc` operator configuration for the corresponding handle.\
`asdBlasCgerc`: multiplies a complex vector by the conjugate transpose of another complex vector, and then adds the result to a matrix.

- Formula:

  $$
  A= alpha * x *y^H + A\\ 
  $$
  Example:\
The input `x` is:\
[1.0 + 1.0j, 2.0 + 2.0j]\
The input `alpha` is:\
1.0 + 1.0j\
The input `y` is:\
[1.0 + 1.0j, 1.0 + 2.0j]\
The input `A` is:\
[ [2.0 + 3.0j, 3.0 + 4.0j], \
  [3.0 + 3.0j, 4.0 + 4.0j] ]\
After the `asdBlasCgerc` operator is called, the output `A` is:\
[ [4.0 + 5.0j, 7.0 + 6.0j], \
  [7.0 + 7.0j, 12.0 + 8.0j] ]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeCgercPlan(
  asdBlasHandle handle)

```

```Cpp
AspbStatus asdBlasCgerc(
  asdBlasHandle               handle, 
  const int64_t               m, 
  const int64_t               n, 
  const std::complex<float> & alpha,
  aclTensor *                 x, 
  const int64_t               incx, 
  aclTensor *                 y, 
  const int64_t               incy, 
  aclTensor *                 A,
  const int64_t               lda)
```

## asdBlasMakeCgercPlan

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

## asdBlasCgerc

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
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of complex elements in vector x, and the number of rows of matrix A.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of complex elements in vector y, and the number of columns of matrix A.</td>
    </tr>
    <tr>
      <td>alpha (std::complex&lt;float&gt; &amp;)</td>
      <td>Input</td>
      <td><ul><li>Alpha in the formula, the input complex scalar.</li><li>Supported data type: <code>COMPLEX64</code>.</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>Input vector, corresponding to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m].</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector x (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>Input vector, corresponding to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr>
    
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements of vector y (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "A" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Indicates the size of the first dimension of matrix A (constrained to m in the current version).</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements `m` and `n` is currently supported in the range [1, 8192].

- The operator input shapes are [m], [n], and [m, n], and the output shape is [m, n].

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
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<float> alpha = std::complex<float>(1.0f, 0.0f);

    int64_t aSize = m * n;
    int64_t xSize = m;
    int64_t ySize = n;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.reserve(aSize);
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.reserve(ySize);

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData[i * n + j] = std::complex<float>(-1.0f, -1.0f);
        }
    }
    for (int64_t i = 0; i < m; i++) {
        tensorInXData[i] = std::complex<float>(1.0 + 1.0 * i, -1.0 - 1.0 * i);
    }
    for (int64_t i = 0; i < n; i++) {
        tensorInYData[i] = std::complex<float>(1.0 + 1.0 * i, -1.0 - 1.0 * i);
    }

    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), 1, m);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, n);

    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> aShape = {m, n};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *inputA = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *inputADeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCgercPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgerc(handle, m, n, alpha, inputX, incx, inputY, incy, inputA, lda));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInAData.data(),
        aSize * sizeof(std::complex<float>),
        inputADeviceAddr,
        aSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy tensor A from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);

    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(inputA);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(inputADeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
