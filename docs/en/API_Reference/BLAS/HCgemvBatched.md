# HCgemvBatched

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:51:55.387Z pushedAt=2026-08-20T11:47:59.756Z -->

## Applicable Product

|Product|Supported|
|:-------------------------|:----------:|
|<term>Atlas 200I/500 A2 inference product</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|√|
|<term>Atlas A2 training products/Atlas A2 inference products</term>|√|
|<term>Ascend 950PR/Ascend 950DT</term>|×|

## Function Description

- API function:

  - `asdBlasMakeHCgemvBatchedPlan`: Initializes the operator configuration corresponding to this handle.

  - `asdBlasHCgemvBatched`: Used to compute the product of batch complex matrices and vectors.

- Formula:

  $$
  y= alpha * op(A)*x + beta * y\\

  $$

  where `op` indicates whether matrix A undergoes a conjugate transpose or non-transpose operation.\
  Example:\
  The input `A[i]` is:\
  [   [ 1+i, 1+2i ],\
    [ 1+3i, 1+4i ]  ]\
  The input `x[i]` is:\
  [ 1+i, 1+i ]\
  The input `trans` is N, indicating that matrix A is not transposed.\
  The input `m` is 2, the input `n` is 2, the input `alpha` is 1+0i, and `beta` is: 0+0i.\
  The input `lda` is: 2.\
  The input `batchCount` is: 2.\
  After the `asdBlasHCgemvBatched` operator is called, the output `y[i]` is:\
  [-1+5i, -5+9i]

## Function Prototype

```Cpp
AspbStatus asdBlasMakeHCgemvBatchedPlan(
  asdBlasHandle        handle, 
  asdBlasOperation_t   trans, 
  const int64_t        m)
```

```Cpp
AspbStatus asdBlasHCgemvBatched(
  asdBlasHandle                    handle, 
  asdBlasOperation_t               trans, 
  const int64_t                    m, 
  const int64_t                    n,
  const std::complex<op::fp16_t> & alpha, 
  aclTensor *                      A, 
  const int64_t                    lda, 
  aclTensor *                      x,
  const int64_t                    incx, 
  const std::complex<op::fp16_t> & beta, 
  aclTensor *                      y, 
  const int64_t                    incy,
  const int64_t                    batchCount)
```

## asdBlasMakeHCgemvBatchedPlan

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
      <td>trans (asdBlasOperation_t)</td>
      <td>Input</td>
      <td>Specifies whether matrix A needs to be transposed.<ul><li><code>ASDBLAS_OP_N</code>: no transpose</li><li><code>ASDBLAS_OP_T</code>: transpose</li><li><code>ASDBLAS_OP_C</code>: conjugate transpose</li></ul></td>
    </tr>
    <tr>
      <td>m (int64_t)</td>
      <td>Input</td>
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## asdBlasHCgemvBatched

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
      <td>Number of rows of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>n (int64_t)</td>
      <td>Input</td>
      <td>Number of columns of matrix A in a single batch.</td>
    </tr>
    <tr>
      <td>alpha (std::complex&lt;op::fp16_t&gt; &amp;)</td>
      <td>Input</td>
      <td>Corresponds to alpha in the formula; a complex scalar used to multiply the result of the matrix-vector multiplication. In the current version, the value of alpha can only be 1+0i.</td>
    </tr>
    <tr>
      <td>A (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "A" in the formula.</li><li>Row-major order.</li><li>Supported data type: <code>COMPLEX32</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batchCount, m, n].</li></ul></td>
    </tr>
    <tr>
      <td>lda (int64_t)</td>
      <td>Input</td>
      <td>Memory address offset between horizontally adjacent elements of A (constrained to m in the current version).</td>
    </tr>
    <tr>
      <td>x (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Input matrix, corresponding to "x" in the formula.</li><li>Supported data type: <code>COMPLEX32</code>.</li><li>Data format: <code>ND</code>.</li><li>When "trans" is non-transpose, the shape is [batchCount, n].</li><li>When "trans" is conjugate transpose, the shape is [batchCount, m].</li></ul></td>
    </tr>
    <tr>
      <td>incx (int64_t)</td>
      <td>Input</td>
      <td>Stride of vector x (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>beta (std::complex&lt;op::fp16_t&gt; &)</td>
      <td>Input</td>
      <td>Corresponds to beta in the formula; a complex scalar used to multiply vector y. In the current version, the value of beta can only be 0+0i.</td>
    </tr>
    <tr>
      <td>y (aclTensor *)</td>
      <td>Input/Output</td>
      <td><ul><li>Corresponds to "y" in the formula.</li><li>Supported data type: <code>COMPLEX32</code>.</li><li>Data format: <code>ND</code>.</li><li>When "trans" is set to non-transpose, the shape is [batchCount, m].</li><li>When "trans" is set to conjugate transpose, the shape is [batchCount, n].</li></ul></td>
    </tr>
    <tr>
      <td>incy (int64_t)</td>
      <td>Input</td>
      <td>Stride of vector y (constrained to 1 in the current version).</td>
    </tr>
    <tr>
      <td>batchCount (int64_t)</td>
      <td>Input</td>
      <td>Number of batches. The value range is [2, 314496].</td>
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

void printTensor(const std::complex<op::fp16_t> *tensorData, int64_t batch, int64_t rows, int64_t cols)
{
    for(int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                auto data = tensorData[b * rows * cols + i * cols + j];
                std::cout << "(" << (float)data.real() << "," << (float)data.imag() << ")" << " ";
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

    // Call the `aclCreateTensor` API to create an `aclTensor`.
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

    int64_t batch = 3;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = m;
    int incx = 1;
    int incy = 1;
    std::complex<op::fp16_t> alpha = std::complex<op::fp16_t>(1.0, 0.0);
    std::complex<op::fp16_t> beta = std::complex<op::fp16_t>(0.0, 0.0);
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    int64_t aSize = batch * m * n;
    int64_t xSize = batch * n;
    int64_t ySize = batch * m;
    std::vector<std::complex<op::fp16_t>> tensorInAData;
    tensorInAData.reserve(aSize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                tensorInAData[b * m * n + i * n + j] = std::complex<op::fp16_t>(i + 0.0f, i + 0.0f);
            }
        }
    }
    std::vector<std::complex<op::fp16_t>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            tensorInXData[b * n + i] = std::complex<op::fp16_t>(i + 1.0f, 2.0f);
        }
    }
    std::vector<std::complex<op::fp16_t>> tensorInYData;
    tensorInYData.reserve(ySize);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            tensorInYData[b * m + i] = std::complex<op::fp16_t>(1.0f, 1.0f);
        }
    }

    std::cout << "trans = " << static_cast<int32_t>(trans) << std::endl;
    std::cout << "alpha = "  << "(" << (float)alpha.real() << "," << (float)alpha.imag() << ")" << std::endl;
    std::cout << "beta = "  << "(" << (float)beta.real() << "," << (float)beta.imag() << ")" << std::endl;
    std::cout << "------- input TensorInA -------" << std::endl;
    printTensor(tensorInAData.data(), batch, m, n);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), batch, 1, n);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), batch, 1, m);

    std::vector<int64_t> aShape = {batch, m, n};
    std::vector<int64_t> xShape = {batch, n};
    std::vector<int64_t> yShape = {batch, m};
    aclTensor *inputA = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    void *inputADeviceAddr = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInAData, aShape, &inputADeviceAddr, aclDataType::ACL_COMPLEX32, &inputA);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX32, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX32, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeHCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    ASD_STATUS_CHECK(asdBlasHCgemvBatched(handle, trans, m, n, alpha, inputA, lda, inputX, incx, beta, inputY, incy, batch));

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    ret = aclrtMemcpy(tensorInYData.data(),
        ySize * sizeof(std::complex<op::fp16_t>),
        inputYDeviceAddr,
        ySize * sizeof(std::complex<op::fp16_t>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- output TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), batch, 1, m);

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
