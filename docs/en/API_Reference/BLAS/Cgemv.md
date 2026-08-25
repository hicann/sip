# Cgemv

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:48:32.268Z pushedAt=2026-08-20T12:19:08.648Z -->

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

`asdBlasMakeCgemvPlan`: initializes the `Cgemv` operator configuration for the corresponding handle.\
`asdBlasCgemv`: a matrix-vector multiplication that computes the product of a complex matrix A and a complex vector x, with the result stored in the complex vector y.

- Formula:

  $$
  y= alpha * op(A)*x + beta * y\\

  $$

  where op(A) = $A\ \text{or}  A^T\  \text{or}\ A^H $, `alpha` and `beta` are scalars, `x` and `y` are vectors, and `A` is an `m*n` matrix.
  Example:
   The input `A` is:\
   [   [ 1+i,1+2i ],
    [ 1+2i,1+3i ]  ]\
   The input `x` is:\
   [ 2+i,2+2i ]\
   The input `y` is: [0+0i]\
   The input `m` is: 2, the input `n` is: 2, the input `trans` is: ASDBLAS_OP_N, the input `alpha` is: 1+i, `beta` is: 2+2i.\
The input `lda` is: 2, the input `incx` is: 1, the input `incy` is: 1.\
Call `asdBlasMakeCgemvPlan` to generate a plan.\
After the `asdBlasCgemv` operator is called, the output `y` is:\
[ -10+8i,-17+9i ]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeCgemvPlan(
  asdBlasHandle        handle, 
  asdBlasOperation_t   trans, 
  const int64_t        m, 
  const int64_t        n,
  aclTensor *          y, 
  const int64_t        incy)
```

```Cpp
AspbStatus asdBlasCgemv(
  asdBlasHandle               handle, 
  asdBlasOperation_t          trans, 
  const int64_t m, 
  const int64_t n,
  const std::complex<float> * alpha, 
  aclTensor *                 A, 
  const int64_t               lda, 
  aclTensor *                 x,
  const int64_t               incx, 
  const std::complex<float> * beta, 
  aclTensor *                 y, 
  const int64_t               incy)
```

## asdBlasMakeCgemvPlan

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
      <td>trans (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A, and number of elements of vector y.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrix A, and number of elements of vector x.</td>
    </tr><tr>
      <td>y (aclTensor *)</td>
      <td>Input</td>
      <td>Vector y.</td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Stride of vector y.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasCgemv

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
      <td>trans (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A, and number of elements of vector y.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrix A, and number of elements of vector x.</td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between adjacent elements in the same row of matrix A (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "A" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m, n].</li></ul></td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Corresponds to "x" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [n].</li></ul></td>
    </tr><tr>
    <td>y (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Input/output matrix, corresponding to "y" in the formula.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [m].</li></ul></td>
    </tr>
    <tr>
      <td>beta (std::complex&lt;float&gt; *)</td>
      <td>Input</td>
      <td>Corresponds to beta in the formula, a complex scalar used to multiply the vector y.</td>
    </tr>
    <tr>
      <td>alpha (std::complex&lt;float&gt; *)</td>
      <td>Input</td>
      <td>Corresponds to alpha in the formula, a complex scalar used to multiply the result of the matrix-vector multiplication.</td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Stride of vector x (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Stride of vector y (constrained to 1 in the current version).</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

- The number of input elements m and n is currently supported in the range [1, 8193].

- The operator input matrix A is in column-major order. The input shapes are [m, n], [m], and [n], and the output shape is [m].

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3 are not supported).

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

void printTensor(const std::complex<float> *tensorData, int64_t rows, int64_t cols)
{
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            std::cout << tensorData[i * cols + j] << " ";
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

    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<float> alpha = std::complex<float>(1.0, 1.0);
    std::complex<float> beta = std::complex<float>(1.0, 1.0);
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    int64_t aSize = m * n;
    int64_t xSize = n;
    int64_t ySize = m;
    std::vector<std::complex<float>> tensorInAData;
    tensorInAData.reserve(aSize);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            tensorInAData[i * n + j] = std::complex<float>(i + 0.0, i + 0.0);
        }
    }
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < n; i++) {
        tensorInXData[i] = std::complex<float>(i + 1.0, 2 + 0.0);
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.reserve(ySize);
    for (int64_t i = 0; i < m; i++) {
        tensorInYData[i] = std::complex<float>(1.0, 1.0);
    }

    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), 1, n);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, m);

    std::vector<int64_t> aShape = {m, n};
    std::vector<int64_t> xShape = {n};
    std::vector<int64_t> yShape = {m};
    aclTensor *inputA = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX64, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeCgemvPlan(handle, trans, m, n, inputY, incy);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasCgemv(handle, trans, m, n, alpha, inputA, lda, inputX, incx, beta, inputY, incy));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInYData.data(),
        ySize * sizeof(std::complex<float>),
        inputYDeviceAddr,
        ySize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), 1, m);

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
