# rsInterpolationBySinc

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:53:58.509Z pushedAt=2026-08-20T11:47:59.776Z -->

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

- API function:

`rsInterpolationBySincGetWorkspaceSize`: Computes the workspace size required by the `rsInterpolationBySinc` operator.\
`rsInterpolationBySinc`: Performs one-dimensional complex vector interpolation with batch support, and returns a result with the same shape and size as the interpolation coordinates.

- Formula:

  $$
  x(d)=\sum _{n=0}^{N-1}x[n]sinc(d-n)
  $$
  where `n` is the real signal index, `x[n]` is the original real signal sequence, and `sinc(d-n)` is the interpolation coefficient, which must be passed as a parameter to this operator.\
  Example:\
The input `inputTensor` is:\
 [ 1 + i , 2 + i ]\
The input `sincTab` is: (intp_num = 2, quant_num = 2)\
  [   [ 1 , 0 ],  [ 0.5 , 0.5 ],   [ 0 , 1 ]  ]\
The original `pos` is:\
  [ 0.2 , 1.6 ] \
Converted to input `posFloor` as: floor(Pos)\
  [ 0 , 1 ] \
Converted to input `posToTabIndex` as: round((Pos -posFloor) × quant_num)\
  [ 0 , 1 ] \
where the tab size is 2*3. Since pos[0] = 0.2, `inputTensor[0]` and the next element `inputTensor[1]`, (a total of 2 elements) are taken, and a dot product is computed with `sincTab[posToTabIndex[0]]` to obtain `outputTensor[0]`. Subsequent elements are computed in sequence.\
  pos[0] = 0.2 → outputTensor[0] = [1 + i , 2 + i] · [ 1 , 0 ] = 1 + i \
  pos[1] = 1.6 → outputTensor[1] = [2 + i , 2 + i] · [ 0.5 , 0.5 ] = 2 + i\
After the `rsInterpolationBySinc` operator is called, the output `outputTensor` is:\
  [ 1 + i , 2 + i ]

## Function Prototype

To use the `rsInterpolationBySinc` operator, first call the `rsInterpolationBySincGetWorkspaceSize` API to obtain the required workspace size and the executor that contains the operator computation flow, and then call the `rsInterpolationBySinc` API to execute the computation.

```Cpp
AspbStatus rsInterpolationBySincGetWorkspaceSize(
  size_t &                   workspaceSize)
```

```Cpp
AspbStatus rsInterpolationBySinc(
  const aclTensor *          inputTensor, 
  const aclTensor *          sincTab,
  const aclTensor *          posFloor, 
  const aclTensor *          posToTabIndex,
  aclTensor *                outputTensor, 
  int                        interpNum, 
  int                        quantNum, 
  int                        interpLength,
  void *                     stream, 
  void *                     workSpace = nullptr)

```

## rsInterpolationBySincGetWorkspaceSize

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
      <td>workspaceSize (size_t &)</td>
      <td>Output</td>
      <td>Address of the workspace.</td>
    </tr>
  </tbody>
    </table>

- **Return value:**

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## rsInterpolationBySinc

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
      <td>inputTensor (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Original signal.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, signalLength].</li></ul></td>
    </tr>
    <tr>
      <td>sincTab (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Interpolation coefficient matrix.</li><li>Supported data type: <code>FLOAT32</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [4, ((quantNum + 1) * 2) * (interpNum * 2 + 8)].</li></ul></td>
    </tr>
    <tr>
      <td>posFloor (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>Value after rounding down the interpolation coordinate.</li><li>Supported data type: <code>INT32</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, length].</li></ul></td>
    </tr>
    <tr>
      <td>posToTabIndex (aclTensor *)</td>
      <td>Input</td>
      <td><ul><li>The row number of the corresponding interpolation coefficient matrix is calculated from the interpolation point coordinates using round((Pos - posFloor) * quantNum).</li><li>Supported data type: <code><code>INT16_T</code></code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, length].</li></ul></td>
    </tr>
    <tr>
      <td>outputTensor (aclTensor *)</td>
      <td>Output</td>
      <td><ul><li>Interpolation result.</li><li>Supported data type: <code>COMPLEX64</code>.</li><li>Data format: <code>ND</code>.</li><li>Shape: [batch, length].</li></ul></td>
    </tr>
    <tr>
      <td>interpNum (int)</td>
      <td>Input</td>
      <td>Number of interpolation points.</td>
    </tr>
    <tr>
      <td>quantNum (int)</td>
      <td>Input</td>
      <td>Number of quantization points.</td>
    </tr>
    <tr>
      <td>interpLength (int)</td>
      <td>Input</td>
      <td>Interpolation length.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>NPU execution stream.</td>
    </tr>
    <tr>
      <td>workSpace (void *)</td>
      <td>Input</td>
      <td>Address of the workspace.</td>
    </tr>
  </tbody>
    </table>

- **Return value**:

  For details about the return values, see [SiP Return Codes](../../context/sip_return_codes.md).

## Constraints

`rsInterpolationBySinc`:

- The number of input elements theoretically ranges from 1 to 3.93e+09.

- During actual computation, the operator does not support high-dimensional ND operations (dimensions ≥ 3).

- `interpNum` only supports even numbers, typically [8, 12, 16]. The maximum supported value in the current version is 16.

- `quantNum` is a power of 2, with a maximum value of 32.

- `inputTensor`, `posFloor`, and `posToTabIndex` have the same number of batches in dimension 0. The length of `outputTensor` is consistent with that of `posFloor` and `posToTabIndex`.

- `sincTab`: To convert complex dot products into real dot products and make the operation more NPU-friendly, preprocessing is required. The table must be expanded to [ ((quantNum+1) × 2) × (interpNum×2+8) * 4]. For the specific algorithm, refer to the preprocessing content in "Calling Example".

## Calling Example

The example code is as follows. This sample is intended to provide a minimal implementation for quick start, development, and debugging of the operator. Its core goal is to demonstrate the core functionality of the operator using the simplest code, rather than providing production-grade security assurance. Users are advised not to directly use the example code for business purposes. If users apply the example code in their own real business scenarios and security issues occur, the users shall bear the consequences themselves.

- `rsInterpolationBySinc` operator calling instructions:

  - The calling example performs sincTab preprocessing:

The coefficient matrix needs to be multiplied with complex numbers. On the NPU, this is converted to multiplying the coefficient matrix with two groups of float32, so the coefficients need to be converted to the following format, where the imaginary matrix is expanded to ((quantNum+1)*2)* (interpNum*2).\
    [coefficient1, 0]\
    [0, coefficient1]

  - For NPU-friendly alignment (32-byte alignment), four matrix formats are required, with 0, 2, 4, or 6 zeros padded at the beginning of each row. The `genTab` function is used to generate coefficient matrices similar to the following:

  w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0,0,0,0,0\
0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0,0,0\
0,0,0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0\
0,0,0,0,0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0

- Call example for the `rsInterpolationBySinc` operator:

```Cpp
#include <iostream>
#include <vector>
#include <securec.h>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using std::complex;
using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0)

#define DINTER_CORE_SIZE 528

static void genTab(float *tab, int tabSize)
{
    static float DINTER_CORE_33x16[DINTER_CORE_SIZE];
    for (int i = 0; i < DINTER_CORE_SIZE; ++i) {
        DINTER_CORE_33x16[i] = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
    }

    for (int i = 0; i < 4; i++) {
        int zeroOffset = i * 2;
        int blockOffset = i * (33 * 2) * (16 * 2 + 8);
        for (int j = 0; j < 33; j++) {
            int rowOffset_real = blockOffset + j * (16 * 2 + 8) * 2;
            int rowOffset_imag = rowOffset_real + (16 * 2 + 8);
            for (int k = 0; k < 16; k++) {
                tab[rowOffset_real + zeroOffset + k * 2] = DINTER_CORE_33x16[j * 16 + k];
                tab[rowOffset_imag + zeroOffset + k * 2 + 1] = DINTER_CORE_33x16[j * 16 + k];
            }
        }
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
    // Copy host-side data to device-side memory using aclrtMemcpy.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int batch = 1;
    int signalLength = 64;
    int interpLength = signalLength;
    const int64_t tabSize = (33 * 2) * (16 * 2 + 8) * 4;  // 2: real and imaginary coefficients, 8: zero padding, 4: no zero padding (pad 2, 4, or 6 zeros).
    const unsigned long inSize = batch * signalLength;
    const unsigned long posSize = batch * interpLength;
    const unsigned long tabIndexSize = batch * interpLength;
    const unsigned long outSize = batch * interpLength;

    float *tabDate = new float[tabSize]();
    genTab(tabDate, tabSize);
    std::vector<float> tab(tabDate, tabDate + tabSize);
    std::vector<complex<float>> inSignal;
    inSignal.reserve(inSize);
    for (long long ii = 0; ii < signalLength; ++ii) {
        inSignal[ii] = complex<float>(ii, ii);
    }
    std::vector<int32_t> intpPos;
    intpPos.reserve(posSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        intpPos[ii] = ii;
    }
    std::vector<int16_t> tabIndex;
    tabIndex.reserve(tabIndexSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        tabIndex[ii] = ii % 33;
    }
    std::vector<complex<float>> outSignal;
    outSignal.reserve(outSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        outSignal[ii] = complex<float>(0, 0);
    }

    aclTensor *tensorIn = nullptr;
    aclTensor *tensorTab = nullptr;
    aclTensor *tensorPos = nullptr;
    aclTensor *tensorTabIndex = nullptr;
    aclTensor *tensorOut = nullptr;
    void *tensorInDeviceAddr = nullptr;
    void *tensorTabDeviceAddr = nullptr;
    void *tensorPosDeviceAddr = nullptr;
    void *tensorTabIndexDeviceAddr = nullptr;
    void *tensorOutDeviceAddr = nullptr;
    ret = CreateAclTensor(inSignal, {batch, signalLength}, &tensorInDeviceAddr, aclDataType::ACL_COMPLEX64, &tensorIn);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tab, {1, tabSize}, &tensorTabDeviceAddr, aclDataType::ACL_FLOAT, &tensorTab);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(intpPos, {batch, interpLength}, &tensorPosDeviceAddr, aclDataType::ACL_INT32, &tensorPos);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        tabIndex, {batch, interpLength}, &tensorTabIndexDeviceAddr, aclDataType::ACL_INT16, &tensorTabIndex);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(outSignal, {batch, interpLength}, &tensorOutDeviceAddr, aclDataType::ACL_COMPLEX64, &tensorOut);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    void *workspace = nullptr;
    size_t workspaceSize = 0;
    rsInterpolationBySincGetWorkspaceSize(workspaceSize);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, static_cast<int64_t>(workspaceSize), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(rsInterpolationBySinc(
        tensorIn, tensorTab, tensorPos, tensorTabIndex, tensorOut, 16, 32, interpLength, stream, workspace));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(outSignal.data(),
        outSize * sizeof(std::complex<float>),
        tensorOutDeviceAddr,
        outSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (long long ii = 0; ii < interpLength; ++ii) {
        std::cout << outSignal[ii] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    delete[] tabDate;
    aclDestroyTensor(tensorIn);
    aclDestroyTensor(tensorPos);
    aclDestroyTensor(tensorTab);
    aclDestroyTensor(tensorTabIndex);
    aclDestroyTensor(tensorOut);
    aclrtFree(tensorInDeviceAddr);
    aclrtFree(tensorTabDeviceAddr);
    aclrtFree(tensorPosDeviceAddr);
    aclrtFree(tensorTabIndexDeviceAddr);
    aclrtFree(tensorOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
