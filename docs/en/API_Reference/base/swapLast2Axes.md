# swapLast2Axes

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:46:52.310Z pushedAt=2026-08-20T11:47:59.715Z -->

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

`swapLast2AxesGetWorkspaceSize`: computes the workspace size required by the `swapLast2Axes` operator.\
`swapLast2Axes`: swaps the last two dimensions of the tensor.

- Formula:

  $$
  outTensor_{bij} = inTensor_{bji}\\
  $$
  Where: *b* is the batch number of the data, *i* is the row number of the input data, and *j* is the column number of the input data.
Example:

  - Example 1:

The input `inTensor` is:\
[[[1.+0.j, 2.+0.j, 3.+0.j]]]\
After the `swapLast2Axes` operator is called, the output `outTensor` is:\
[[[1.+0.j], [2.+0.j], [3.+0.j]]]

  - Example 2:

The input `inTensor` is:\
[[[ 0.+0.j, 1.+0.j, 2.+0.j], \
[ 3.+0.j, 4.+0.j, 5.+0.j]],\
[[ 6.+0.j, 7.+0.j, 8.+0.j], \
[ 9.+0.j, 10.+0.j, 11.+0.j]]]\
After the `swapLast2Axes` operator is called, the output `outTensor` is:\
[[[ 0.+0.j, 3.+0.j], \
[ 1.+0.j, 4.+0.j], \
[ 2.+0.j, 5.+0.j]],\
[[ 6.+0.j, 9.+0.j], \
[ 7.+0.j, 10.+0.j], \
[ 8.+0.j, 11.+0.j]]]

## Function Prototype

To use the `swapLast2Axes` operator, the `swapLast2AxesGetWorkspaceSize` API must be called first to obtain the input parameters and compute the required workspace size based on the computation process, and then the `swapLast2Axes` API is called to perform the computation.

```Cpp
AsdSip::AspbStatus swapLast2AxesGetWorkspaceSize(
  size_t *size)
```

```Cpp
AsdSip::AspbStatus swapLast2Axes(
  const aclTensor *    inTensor, 
  aclTensor *          outTensor, 
  void *               stream,
  void *               workspace = nullptr)
```

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## swapLast2AxesGetWorkspaceSize

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
      <td>size (size_t *)</td>
      <td>Input/Output</td>
      <td>Workspace required by the <code>swapLast2Axes</code> operator.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## swapLast2Axes

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
      <td>inTensor (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Indicates the input tensor data, corresponding to "inTensor" in the formula.</li><li>The maximum number of input elements is 3,600,000,000 (within [60000, 60000]).</li><li>Only <code>COMPLEX64</code> is supported. Data format: <code>ND</code>.</li>
      <li>The input dimension is limited to 2 or 3.</li></ul>
      </td>
    </tr>
    <tr>
      <td>outTensor (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Indicates the output tensor data, corresponding to "outTensor" in the formula.</li><li>Only <code>COMPLEX64</code> is supported. The data type must be consistent with that of <code>inTensor</code>.</li><li>If the shape of <code>inTensor</code> is [k, x, y], the shape of <code>outTensor</code> is [k, y, x].</li><li>Data format: <code>ND</code>.</li></ul></td>
    </tr>
    <tr>
      <td>workspace (void *)</td>
      <td>Input</td>
      <td>Workspace required by the <code>swapLast2Axes</code> operator.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>NPU execution stream.</td>
    </tr>
  </tbody>
  </table>

- **Return value:**

For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

During actual computation, the operator does not support high-dimensional ND operations (dimensions > 3).

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
    auto size = GetShapeSize(shape) * sizeof(T) * 2;
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

void printTensor(const float *tensorData, size_t row, size_t col)
{
    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < col; ++c) {
            size_t index = (r * col + c) * 2;
            std::cout << "(" << int(tensorData[index]) << ", " << int(tensorData[index + 1]) << ") ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t row = 3;
    int64_t col = 2;
    const int64_t tensorSize = row * col * 2;

    std::vector<float> tensorInData;
    tensorInData.reserve(tensorSize);
    for (int64_t i = 0; i < tensorSize; i++) {
        tensorInData[i] = 0.0 + i;
    }
    std::vector<float> tensorOutData;
    tensorOutData.reserve(tensorSize);

    std::vector<int64_t> inShape = {row, col};
    std::vector<int64_t> outShape = {col, row};
    aclTensor *input = nullptr;
    aclTensor *output = nullptr;
    void *inputDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutData, outShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &output);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    void *workspace = nullptr;
    size_t lwork = 0;
    swapLast2AxesGetWorkspaceSize(lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&workspace, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(swapLast2Axes(input, output, stream, workspace));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutData.data(),
        tensorSize * sizeof(float),
        outputDeviceAddr,
        tensorSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy output tensor from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "row = " << row << ", col = " << col << std::endl;
    std::cout << "------- Input ------- " << std::endl;
    printTensor(tensorInData.data(), row, col);

    std::cout << "------- Output -------" << std::endl;
    printTensor(tensorOutData.data(), col, row);
    std::cout << "Execute successfully." << std::endl;

    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclDestroyTensor(input);
    aclDestroyTensor(output);
    if (lwork > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
