# FFT_2D

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √  |

## 功能说明

- 接口功能：\
asdFftMakePlan2D：初始化二维FFT配置。\
asdFftExecC2C：执行复数到复数的FFT变换。\
asdFftExecC2R：执行复数到实数的FFT变换。\
asdFftExecR2C：执行实数到复数的FFT变换。
- 计算公式：\
傅里叶变换（Fourier transform）是一种线性积分变换，用于信号在时域和频域之间的变换，在物理学和工程学中有许多应用。对应给定长度为N的信号，其离散形式DFT(Discrete Fourier Transform)表达式如下：

  ![公式](../figures/FFT_2D_1.png)

  将系数矩阵(N*N)和时域信号(N*1)看做两个Tensor，在NPU上直接使用矩阵乘，可完成DFT，但时间复杂度太高，因此需要快速傅里叶变换。其基本原理是利用三角函数在复数域的旋转对称性，将序列拆分成子序列，通过蝶形运算以降低计算的复杂度：\
  ![公式](../figures/FFT_ID_2.png)

## 函数原型

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

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>算子的句柄，需要手动申请创建asdFftHandle对象。</td>
    </tr>
    <tr>
      <td>fftSizeX（int64_t）</td>
      <td>输入</td>
      <td>对应公式中的'M'，FFT信号长度（第一维）。</td>
    </tr>
    <tr>
      <td>fftSizeY（int64_t）</td>
      <td>输入</td>
      <td>对应公式中的'N'，FFT信号长度（第二维）。</td>
    </tr>
    <tr>
      <td>fftType（asdFftType）</td>
      <td>输入</td>
      <td>FFT变换类型<ul><li>ASCEND_FFT_C2C：复数到复数的快速傅里叶变换。</li><li>ASCEND_FFT_C2R：复数到实数的快速傅里叶变换。</li><li>ASCEND_FFT_R2C：实数到复数的快速傅里叶变换。</li></ul></td>
    </tr>
    <tr>
      <td>direction（asdFftDirection）</td>
      <td>输入</td>
      <td>选择FFT执行正向变换或反向变换<ul><li>ASCEND_FFT_FORWARD：正向快速傅里叶变换。</li><li>ASCEND_FFT_INVERSE：逆向快速傅里叶变换。</li></ul></td>
    </tr>
    <tr>
      <td>batchSize（int32_t）</td>
      <td>输入</td>
      <td>FFT变换批处理操作中的数据批次数量。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。
  
## asdFftExecC2C

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>算子的句柄，需要手动申请创建asdFftHandle对象。</td>
    </tr>
    <tr>
      <td>input（ const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>输入的shape为（batchSize，fftSizeX，fftSizeY）。</li></ul></td>
    </tr>
    <tr>
      <td>output（const aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'y'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>输出的shape为（batchSize，fftSizeX，fftSizeY）。</li></ul></td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdFftExecC2R

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>算子的句柄，需要手动申请创建asdFftHandle对象。</td>
    </tr>
    <tr>
      <td>input（ const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>输入的shape为（batchSize，fftSizeX，fftSizeY/2+1）。</li></ul></td>
    </tr>
    <tr>
      <td>output（const aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'y'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>输出的shape为（batchSize，fftSizeX，fftSizeY）。</li></ul></td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdFftExecR2C

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdFftHandle）</td>
      <td>输入</td>
      <td>算子的句柄，需要手动申请创建asdFftHandle对象。</td>
    </tr>
    <tr>
      <td>input（ const aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'x'。</li><li>数据类型支持FLOAT32。</li><li>数据格式支持ND。</li>
      <li>输入的shape为（batchSize，fftSizeX，fftSizeY）。</li></ul></td>
    </tr>
    <tr>
      <td>output（const aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'y'。</li><li>数据类型支持COMPLEX64。</li><li>数据格式支持ND。</li>
      <li>输出的shape为（batchSize，fftSizeX，fftSizeY/2+1）。</li></ul></td>
    </tr>
    </tbody>
    </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

asdFftMakePlan2D

  - fftSizeX、fftSizeY需保证不超过$2^{27}$且分解质因数后不包含超过199的质因子。
  - batchSize在存储允许范围内应无额外约束。
  - 输入的元素个数理论支持[1，$2^{30}$]。
  - 输入的元素不支持inf、-inf和nan，如果输入中包含这些值， 那么结果为未定义。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

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
    // 固定写法，AscendCL初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
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

    // 创造tensor的Host侧数据
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

    // 打印输出tensor值中前16个
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
    // 固定写法，AscendCL初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
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

    // 创造tensor的Host侧数据
    // int batch = 32, Nfft = 128;
    // int batch = 32, Nfft = 8192;
    // int batch = 32, Nfft = 15000;

    // 创造tensor的Host侧数据
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

    // 打印输出tensor值中前16个
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
    // 固定写法，AscendCL初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
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

    // 创造tensor的Host侧数据
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

    // 打印输出tensor值中前16个
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
